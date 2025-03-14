import os
import time
import torch
import torch.optim as optim
import multiprocessing as mp
from datetime import timedelta

from lib.a3c.model import ActorCritic
from lib.a3c.agent import worker_process
from helpers.config import load_config
from helpers.logger import Logger


def train_a3c():
    """Train the A3C agent."""
    # Load configuration
    config = load_config("a3c")
    device = torch.device(config["device"])

    # Initialize logger
    logger = Logger(config, "A3C")

    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # TODO: Create the global network
    # Hint: Use the ActorCritic class with parameters from config
    # The network should be moved to the specified device and shared among processes
    action_dim = config["network"].get("action_dim", 1)
    global_net = ActorCritic(config["network"]["state_size"], action_dim, config).to(device)
    global_net = global_net.share_memory()

    # TODO: Create optimizer for the global network
    # Hint: Use Adam optimizer with learning rate from config
    optimizer = optim.Adam(global_net.parameters(), lr=config["hyperparameters"]["lr"])

    # TODO: Create global episode counter and lock for synchronization
    # Hint: Use mp.Value for the counter and mp.Lock for the lock
    global_ep = mp.Value("i", 0)
    lock = mp.Lock()

    # Create directory for saving models
    os.makedirs(config["logging"]["model_dir"], exist_ok=True)

    # Start training
    logger.project_logger.info("Starting A3C training for Kuka pick and place task...")
    logger.project_logger.info(
        f"Using {config['hyperparameters']['num_workers']} workers on {device}"
    )
    start_time = time.time()

    # Get log and model directories from logger
    log_dir = logger.log_dir
    model_dir = logger.model_dir

    # TODO: Create and start worker processes
    # Hint: Use mp.Process to create workers and pass necessary arguments to worker_process
    # Only worker 0 should get the log and model directories
    processes = []
    for worker_id in range(config["hyperparameters"]["num_workers"]):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                global_net,
                optimizer,
                global_ep,
                config["hyperparameters"]["max_episodes"],
                lock,
                config,
                device,
                log_dir if worker_id == 0 else None,
                model_dir if worker_id == 0 else None,
            ),
        )
        p.start()
        processes.append(p)

    # TODO: Wait for all processes to finish
    for p in processes:
        p.join()

    training_time = time.time() - start_time
    logger.project_logger.info(
        f"Training completed in {timedelta(seconds=int(training_time))}"
    )

    # TODO: Save the final trained model
    # Hint: Use torch.save to save the model state dict, optimizer state dict, and episode count
    model_path = os.path.join(model_dir, "final_a3c_model.pth")
    torch.save(
        {
            "model_state_dict": global_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_episode": global_ep.value,
        },
        model_path,
    )

    logger.project_logger.info(f"Final model saved to {model_path}. Training complete!")
    logger.close()
