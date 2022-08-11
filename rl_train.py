import os
from glob import glob
from rl_environment import AptamerEnvironment
from newAgent import SelectionPolicyAgent
import torch
from comet_ml import Experiment
import numpy as np
from utils import printRecord
from pathlib import Path

"""
This Script will Train a Reinforcement Learning Agent.

It will:
1)

Runs
    Run{n}
        Episode{x}
            ckpts
                ...
            datasets
                ...
            outputsDict.npy
            record
        Episode{x+1}
            ckpts
                ...
            datasets
                ...
            outputsDict.npy
            record
        ...
    Run{n+1}
        ...

"""


def trainRLAgent(config):
    workdir, _ = makeNewWorkingDirectory(
        config.workdir,
        explicit_run_enumeration=config.explicit_run_enumeration,
        runNum=config.run_num,
    )
    config.workdir = workdir  # I'M RESETTING the workdir CONFIG PROPERTY
    logger = setupLogger(config, workdir)
    agent = SelectionPolicyAgent(config)
    env = AptamerEnvironment(config, logger, workdir)
    torch.manual_seed(config.rl.seed)

    trial_episode_scores = []
    avg_grad_value = None
    debug_log = True

    for i_episode in range(config.rl.episodes):
        episode_score = 0
        state, _ = env.reset()
        agent.updateState(state)

        # agent.updateState(state, proxy_model)
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            action = agent.select_action()
            new_state, reward, done, _ = env.step(agent.action_to_map(action))
            episode_score += reward

            # save state, action, reward sequence
            agent.memory.push(state, action, new_state, reward, done)
            state = new_state
            agent.updateState(state)

        if (
            len(agent.memory) >= config.rl.min_memory
            and (i_episode % config.rl.dqn_train_frequency == 0)
            and i_episode >= config.rl.learning_start
        ):
            agent.policy_error = []
            agent.train()
            agent.update_target_network()
            avg_grad_value = torch.mean(
                torch.stack(
                    [torch.mean(abs(param.grad)) for param in agent.policy_net.parameters()]
                )
            )

        avg_param_value = torch.mean(
            torch.stack([torch.mean(abs(param)) for param in agent.policy_net.parameters()])
        )

        # Update epsilon and learning rate
        agent.scheduler.step()
        agent.update_epsilon()

        # Logging
        trial_episode_scores += [episode_score]
        last_100_avg = np.mean(trial_episode_scores[-100:])
        if debug_log:
            if avg_grad_value:
                print(
                    f"E {i_episode} scored {episode_score:.2f}, ep_length {episode_length}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}",
                    end=", ",
                )
                print(f"avg_grad {avg_grad_value:.2f} avg_loss {np.mean(agent.policy_error):.2f}")
            else:
                print(
                    f"E {i_episode} scored {episode_score:.2f}, ep_length {episode_length}, avg {last_100_avg:.2f}, avg param {avg_param_value:.2f}"
                )
        if i_episode % config.rl.eval_interval == 0:
            mean_eval_score = agent.evaluate(env)
            logger.log_metric(name="Mean Eval Score", value=mean_eval_score, step=i_episode)

        logger.log_metric(name="Learning Rate", value=agent.scheduler.get_last_lr(), step=i_episode)
        logger.log_metric(name="Episode Duration", value=episode_length, step=i_episode)
        logger.log_metric(name="Episode Score", value=episode_score, step=i_episode)
        logger.log_metric(name="Moving Score Average (100 eps)", value=last_100_avg, step=i_episode)
        logger.log_metric(name="Mean Gradient Value", value=avg_grad_value, step=i_episode)
        logger.log_metric(name="Mean Weight Value", value=avg_param_value, step=i_episode)
        logger.log_metric(
            name="Mean Training Replay Loss", value=np.mean(agent.policy_error), step=i_episode
        )


# Train Policy Network


def endOfEpisode(agent, comet=None):
    agent.handleTraining()
    agent.handleEndofEpisode(logger=comet)
    if comet:
        comet.log_metric(
            name="RL Cumulative Reward",
            value=model_state_cumulative_reward,
            step=agent.episode,
        )
        comet.log_metric(
            name="RL Cumulative Score",
            value=model_state_cumulative_score,
            step=agent.episode,
        )
        comet.log_metric(
            name="RL Dataset Cumulative Score",
            value=dataset_cumulative_score,
            step=agent.episode,
        )
    if config.rl.episodes > (agent.episode + 1):  # if we are doing multiple al episodes
        agent.episode += 1
        reset()

    agent.save_models()


def makeNewWorkingDirectory(
    rundir, explicit_run_enumeration=False, runNum=None
):  # make working directory
    """
    make a new working directory
    non-overlapping previous entries
    :return:
    """
    if explicit_run_enumeration and runNum:
        newdir = f"{rundir}/run{runNum}"
        os.mkdir(newdir)
    else:
        rundirs = glob(rundir + "/" + "run*")  # check for prior working directories
        if len(rundirs) > 0:
            prev_runs = []
            for i in range(len(rundirs)):
                prev_runs.append(int(rundirs[i].split("run")[-1]))

            prev_max = max(prev_runs)
            runNum = prev_max + 1
            newdir = rundir + "/" + "run%d" % (runNum)
            os.mkdir(newdir)
        else:
            runNum = 1
            newdir = f"{rundir}/run{runNum}"
            os.mkdir(newdir)

    os.mkdir(f"{newdir}/datasets")
    os.mkdir(f"{newdir}/ckpts")
    return newdir, runNum


def setupLogger(config, workdir):
    logger = Experiment(
        project_name=config.al.comet.project,
        display_summary_level=0,
    )
    if config.al.comet.tags:
        if isinstance(config.al.comet.tags, list):
            logger.add_tags(config.al.comet.tags)
        else:
            logger.add_tag(config.al.comet.tags)
    hyperparams = vars(config.al)
    hyperparams.pop("comet")
    logger.log_parameters(vars(config.rl))
    logger.set_name("run {}".format(config.run_num))
    with open(Path(workdir) / "comet_al.url", "w") as f:
        f.write(logger.url + "\n")
