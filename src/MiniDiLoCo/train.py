import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from MiniDiLoCo.aggregator import AllReduce
from torch.utils.data import DistributedSampler, DataLoader
import os, sys, json
from .utils import setup, cleanup
from .strategy import Diloco
import matplotlib.pyplot as plt
from pathlib import Path
import time


class Trainer:
    """ Generic distributed PyTorch trainer """

    def __init__(self, rank, world_size, model, dataloader):
        self.rank = rank
        self.world_size = world_size
        #self.device = torch.device(f"cuda:{rank}")
        self.device = torch.device(f"cuda:2")
        self.model = DDP(model.to(self.device), device_ids=[self.device.index])
        self.dataloader = dataloader


    def train(self, strategy, total_steps):
        """ Distributed outer training loop.

        Args:
            strategy (Strategy): Distributed strategy chooses who performs the inner loop
            total_steps (Int): Number of outer loops
        """        

        strategy._init_node(self.model, self.rank, self.world_size, total_steps)
        all_reduce = AllReduce(strategy.outer_opt, strategy.inner_opt)
        
        self.model.train()
        loss_history = []

        print(f"TAILLE DU DATALOADER: {len(self.dataloader)}")
        global_loss_for_plot = []
        start_time = time.time()  # TIME
        max_hours = 5  # TIME

        for outer_step in range(total_steps):
            self.dataloader.sampler.set_epoch(outer_step)

            running_loss = 0.0
            loss_for_plot = []  # FOR THE PLOT

            data_iter = iter(self.dataloader)
            for inner_step in range(strategy.H):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Inner loop
                loss = strategy.step(batch)
                loss_for_plot.append(loss) # FOR THE PLOT
                running_loss += loss

            global_loss_for_plot.append(loss_for_plot) # FOR THE PLOT
            loss_history.append(running_loss / strategy.H)

            strategy.outer_opt.zero_grad()
            # Parameters synchronisation
            all_reduce.aggregate()

            strategy.outer_opt.step()
            strategy.scheduler.step()

            # Resync inner params
            outer_params_iter = (p for group in strategy.outer_opt.param_groups for p in group['params'])
            for outer_p, inner_p in zip(outer_params_iter, self.model.parameters()):
                inner_p.data.copy_(outer_p.data)

            #strategy.inner_opt.state.clear()

            ###########

            #for state in strategy.inner_opt.state.values():
            #    if 'exp_avg' in state:
            #        state['exp_avg'].zero_()  # ← reset m_t (direction)
            #    if 'step' in state:
            #        state['step'] = torch.zeros_like(state['step'])

            #from transformers import get_cosine_schedule_with_warmup
            #strategy.scheduler = get_cosine_schedule_with_warmup(
            #    strategy.inner_opt,
            #    num_warmup_steps=strategy.warmup_steps,
            #    num_training_steps=strategy.H
            #)

            #print(f"Rank: {self.rank} - Outer step: {outer_step} - Average loss: {running_loss / strategy.H}")
            print(f"Rank: {self.device.index} - Outer step: {outer_step} - Average loss: {running_loss / strategy.H}")

            if time.time() - start_time >= max_hours * 3600:
                print(f"Rank: {self.rank} - Arrêt après {max_hours}h") # TIME
                break

        return loss_history, global_loss_for_plot

def make_worker_dataloader(rank, world_size, dataset, batch_size, mode="chunked"):
    if mode == "chunked":
        videos, masks = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            chunks_v = torch.chunk(sample["pixel_values"], world_size, dim=3)
            chunks_m = torch.chunk(sample["labels"],       world_size, dim=3)

            if i == 0:
                video_full = sample["pixel_values"]
                mask_full  = sample["labels"]
                mid = video_full.shape[1] // 2

                fig, axes = plt.subplots(2, world_size + 1, figsize=(4 * (world_size + 1), 8))

                axes[0, 0].imshow(video_full[0, mid].numpy(), cmap='gray')
                axes[0, 0].set_title(f"[FULL] Video | shape: {tuple(video_full.shape)}")
                axes[0, 0].axis('off')

                axes[1, 0].imshow(mask_full[0, mid].numpy(), cmap='gray')
                axes[1, 0].set_title(f"[FULL] Mask | shape: {tuple(mask_full.shape)}")
                axes[1, 0].axis('off')

                for r in range(world_size):
                    v_chunk = chunks_v[r]
                    m_chunk = chunks_m[r]

                    axes[0, r + 1].imshow(v_chunk[0, mid].numpy(), cmap='gray')
                    axes[0, r + 1].set_title(f"[chunk {r}] Video | shape: {tuple(v_chunk.shape)}")
                    axes[0, r + 1].axis('off')

                    axes[1, r + 1].imshow(m_chunk[0, mid].numpy(), cmap='gray')
                    axes[1, r + 1].set_title(f"[chunk {r}] Mask | shape: {tuple(m_chunk.shape)}")
                    axes[1, r + 1].axis('off')

                plt.tight_layout()

                Path("plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"plots/verify_chunks_rank{rank}.png", dpi=150)
                plt.close()

            videos.append(chunks_v[rank].clone())
            masks.append(chunks_m[rank].clone())

        videos = torch.stack(videos)
        masks  = torch.stack(masks)

        class ChunkDataset(torch.utils.data.Dataset):
            def __init__(self, v, m): self.v, self.m = v, m
            def __len__(self): return len(self.v)
            def __getitem__(self, idx): return {"pixel_values": self.v[idx], "labels": self.m[idx]}

        worker_dataset = ChunkDataset(videos, masks)
        sampler = DistributedSampler(worker_dataset, num_replicas=1, rank=0, shuffle=True)

    elif mode == "full":
        worker_dataset = dataset
        sampler = DistributedSampler(worker_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    else:
        raise ValueError(f"mode doit être 'chunked' ou 'full', reçu : {mode!r}")

    return DataLoader(worker_dataset, batch_size=batch_size, sampler=sampler)

def diloco_worker(rank, world_size, model, dataset, batch_size, loss_fn,
                  num_epochs, lr, save_path,
                  inner_optimizer_cls=None, inner_optimizer_kwargs=None,
                  outer_optimizer_cls=None,  outer_optimizer_kwargs=None,
                  mode="full"):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK']  = str(rank)
    os.environ['WORLD_SIZE']  = str(world_size)

    sys.path.insert(0, os.path.dirname(__file__))
    setup('nccl', rank, world_size)

    dataloader = make_worker_dataloader(rank, world_size, dataset, batch_size, mode)

    strategy = Diloco(
        loss_fn=loss_fn,
        inner_optimizer_cls=inner_optimizer_cls,
        inner_optimizer_kwargs=inner_optimizer_kwargs or {"lr": lr},
        outer_optimizer_cls=outer_optimizer_cls,
        outer_optimizer_kwargs=outer_optimizer_kwargs,
        H=464,
    )
    trainer = Trainer(rank, world_size, model, dataloader)
    loss_history, global_loss_for_plot = trainer.train(strategy, total_steps=num_epochs)

    # ── Vérification avant sauvegarde ──────────────────────────────────────
    raw_model = model.module if hasattr(model, 'module') else model

    first_param = next(raw_model.parameters())
    outer_params = [p for group in strategy.outer_opt.param_groups for p in group['params']]

    print(f"[rank {rank}] model.module clé : {hasattr(model, 'module')}")
    print(f"[rank {rank}] Premier param du modèle (sum): {first_param.sum().item():.6f}")
    print(f"[rank {rank}] Premier outer_param (sum):     {outer_params[0].sum().item():.6f}")
    print(f"[rank {rank}] Sont-ils identiques ? {torch.allclose(first_param, outer_params[0])}")
    # ────────────────────────────────────────────────────────────────────────

    if rank == 0:
        ##### FOR THE PLOT #####
        #for i in range(len(global_loss_for_plot)):
        #    loss_for_plot = global_loss_for_plot[i]
        #    print(f"size: {len(loss_for_plot)}")
        #    print(loss_for_plot)
        #
        #   n_points = len(loss_for_plot)
        #    n_bins = 100  # nombre d'intervalles pour la moyenne (ajustable)
        #    bin_size = max(n_points // n_bins, 1)
        #
        #    n_full_bins = n_points // bin_size
        #    trimmed = loss_for_plot[:n_full_bins * bin_size]
        #    binned_means = np.array(trimmed).reshape(-1, bin_size).mean(axis=1)
        #    bin_centers = np.arange(n_full_bins) * bin_size + bin_size / 2
        #
        #    plt.figure(figsize=(16, 6))
        #
        #    plt.plot(loss_for_plot, color="steelblue", alpha=0.3, linewidth=0.5, label="Toutes les valeurs (5000 pts)")
        #
        #    plt.plot(bin_centers, binned_means, color="darkorange", linewidth=2, marker="o", markersize=4,
        #             label=f"Moyenne par intervalle de {bin_size} pts")
        #
        #    plt.axhline(
        #        y=np.average(loss_history[i]),
        #        color="red",
        #        linestyle="--",
        #        linewidth=2,
        #        label=f"Mean loss globale = {np.average(loss_history[i]):.6f}"
        #    )
        #
        #    plt.ylabel("Loss values")
        #    plt.xlabel("Iterations")
        #    plt.title(f"Previous average loss {loss_history[i - 1] if i > 0 else 'None'}")
        #    plt.legend()
        #    plt.tight_layout()
        #    plt.savefig(f"plots/loss_step_{i}.png", dpi=150)
        #    plt.close()
        ########################

        torch.save(raw_model.state_dict(), save_path)
        #torch.save(model.state_dict(), save_path)

        loss_path = save_path.replace(".pth", "_loss.json")
        with open(loss_path, "w") as f:
            json.dump(loss_history, f)

        print(f"[rank 0] Saved → {save_path}")

    torch.distributed.barrier()
    cleanup()