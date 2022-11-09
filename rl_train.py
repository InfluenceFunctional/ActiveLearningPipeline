import os
from glob import glob
from rl_environment import AptamerEnvironment
from newAgent import SelectionPolicyAgent
import torch
from comet_ml import Experiment
import numpy as np
from pathlib import Path
import time
import copy

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


def getBaseline(config, rundir):
    """Runs the ActiveLearningPipeline with query_mode set to heuristic but
    outputs performance metrics like the evaluation pipeline."""
    trial_episode_scores = []
    trial_model_state_score = []
    trial_dataset_score = []

    baseline_config = copy.deepcopy(config)
    baseline_config.al.query_mode = "heuristic"
    env = AptamerEnvironment(baseline_config, rundir=rundir)
    torch.manual_seed(baseline_config.rl.seed)

    for i_episode in range(1):
        episode_score = 0
        _, _ = env.reset(i_episode)
        done = False
        while not done:
            _, reward, done, env_info = env.step(None)
            episode_score += reward

        trial_episode_scores += [episode_score]
        trial_model_state_score += [env_info["model_state_cumul_score"]]
        trial_dataset_score += [env_info["dataset_cumul_score"]]
    return (
        np.mean(trial_episode_scores),
        np.mean(trial_model_state_score),
        np.mean(trial_dataset_score),
    )


def get_normalized_actions(actions):
    tradeoff = 0
    tradeoffs = []
    binary_to_policy = np.array(((1, 1, 1, 0, 0, 0, -1, -1, -1), (1, 0, -1, 1, 0, -1, 1, 0, -1)))

    c1 = [0.5]
    c2 = [0.5]
    for action_id in actions:
        action_map = torch.zeros(9, dtype=int)
        action_map[action_id] = 1
        # action 1 is for dist cutoff modulation, action 2 is for c1-c2 tradeoff
        action = binary_to_policy @ np.asarray(action_map)
        tradeoff += action[1] * 0.1  # modulate by 0.1
        tradeoffs += [tradeoff]
        c1 += [0.5 - tradeoff / 2]
        c2 += [0.5 + tradeoff / 2]

    return c1, c2, tradeoffs  # , c1_norm, c2_norm


def trainRLAgent(config):

    rundir, _ = makeNewWorkingDirectory(
        config.workdir,
        explicit_run_enumeration=config.explicit_run_enumeration,
        runNum=config.run_num,
    )
    config.workdir = rundir  # I'M RESETTING the workdir CONFIG PROPERTY
    logger = setupLogger(config, rundir)

    if config.rl.calculate_baseline:
        baseline_folder_dir = f"{rundir}/Baseline/"
        os.mkdir(baseline_folder_dir)
        baseline_episode_score, baseline_model_state_score, baseline_dataset_score = getBaseline(
            config, rundir=f"{rundir}/Baseline"
        )
        logger.log_metric(value=baseline_episode_score, name="Baseline Episode Score")
        logger.log_metric(value=baseline_model_state_score, name="Baseline Model State Score")
        logger.log_metric(value=baseline_dataset_score, name="Baseline Dataset Score")

    agent = SelectionPolicyAgent(config)
    env = AptamerEnvironment(config, logger, rundir)
    torch.manual_seed(config.rl.seed)

    trial_episode_scores = []
    avg_grad_value = None

    for i_episode in range(config.rl.episodes):
        t0 = time.time()
        episode_score = 0
        state, _ = env.reset(i_episode)
        agent.updateState(state)
        actions = []

        # agent.updateState(state, proxy_model)
        done = False
        episode_length = 0
        while not done:
            episode_length += 1
            action = agent.select_action()
            actions += [action]
            new_state, reward, done, env_info = env.step(agent.action_to_map(action))
            episode_score += reward

            # save state, action, reward sequence
            agent.memory.push(state, action, new_state, reward, done)
            state = new_state
            agent.updateState(state)

        # Train Agent if conditions are met.
        agent.handleTraining(i_episode)

        avg_param_value = torch.mean(
            torch.stack([torch.mean(abs(param)) for param in agent.policy_net.parameters()])
        )

        # Update epsilon and learning rate
        agent.update_epsilon()
        agent.scheduler.step()
        # Logging
        trial_episode_scores += [episode_score]
        last_100_avg = np.mean(trial_episode_scores[-100:])

        print(f"Episode {i_episode} took {int(time.time() - t0)} seconds\n")
        if config.rl.log_rl_to_console:
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

        if i_episode % config.rl.eval_interval == 0 and i_episode != 0:
            mean_eval_score = agent.evaluate(env, config.rl.eval_steps)
            logger.log_metric(name="Mean Eval Score", value=mean_eval_score, step=i_episode)
            c1, c2, _ = get_normalized_actions(actions)
            logger.log_curve(
                name=f"Ep {i_episode} - Energy",
                x=range(10),
                y=c1,
                step=i_episode,
            )

        # logger.log_curve(
        #    name=f"Ep {i_episode} - Uncertainty",
        #    x=list(range(10)),
        #    y=c2,
        #    step=i_episode,
        # )
        logger.log_metric(name="Learning Rate", value=agent.scheduler.get_last_lr(), step=i_episode)
        logger.log_metric(name="Episode Duration", value=episode_length, step=i_episode)
        logger.log_metric(name="Episode Score", value=episode_score, step=i_episode)
        logger.log_metric(name="Moving Score Average (100 eps)", value=last_100_avg, step=i_episode)
        logger.log_metric(name="Mean Gradient Value", value=avg_grad_value, step=i_episode)
        logger.log_metric(name="Mean Weight Value", value=avg_param_value, step=i_episode)
        logger.log_metric(
            name="Mean Training Replay Loss", value=np.mean(agent.policy_error), step=i_episode
        )
        logger.log_metric(
            name="Model State Cumulative Score",
            value=env_info["model_state_cumul_score"],
            step=i_episode,
        )
        logger.log_metric(
            name="Dataset Cumulative Score", value=env_info["dataset_cumul_score"], step=i_episode
        )

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
    return logger
