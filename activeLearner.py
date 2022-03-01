from comet_ml import Experiment
from argparse import Namespace
import comet_ml
import yaml
from models import modelNet
from querier import *
from sampler import *
from utils import namespace2dict
from torch.utils import data
import torch.nn.functional as F
import torch
from Agent import ParameterUpdateAgent
from replay_buffer import ParameterUpdateReplayMemory
import pandas as pd

import numpy
import os
import glob
import multiprocessing as mp


class ActiveLearning:
    def __init__(self, config):
        self.pipeIter = None
        self.homedir = os.getcwd()
        self.episode = 0
        self.config = config
        self.runNum = self.config.run_num
        self.oracle = Oracle(
            self.config
        )  # oracle needs to be initialized to initialize toy datasets
        self.agent = ParameterUpdateAgent(self.config)
        self.querier = Querier(self.config)  # might as well initialize the querier here
        self.setup()
        self.getModelSize()
        # Comet
        if config.al.comet.project:
            self.comet = Experiment(project_name=config.al.comet.project, display_summary_level=0,)
            if config.al.comet.tags:
                if isinstance(config.al.comet.tags, list):
                    self.comet.add_tags(config.al.comet.tags)
                else:
                    self.comet.add_tag(config.al.comet.tags)

            self.comet.set_name("run {}".format(config.run_num))

            self.comet.log_parameters(vars(config))
            with open(Path(self.workDir) / "comet_al.url", "w") as f:
                f.write(self.comet.url + "\n")
        else:
            self.comet = None
        # Save YAML config
        with open(self.workDir + "/config.yml", "w") as f:
            yaml.dump(numpy2python(namespace2dict(self.config)), f, default_flow_style=False)

    def setup(self):
        """
        setup working directory
        move to relevant directory
        :return:
        """
        if self.config.run_num == 0:  # if making a new workdir
            self.makeNewWorkingDirectory()
            self.reset()
        elif self.config.explicit_run_enumeration == True:
            self.workDir = (
                self.config.workdir + "/run%d" % self.config.run_num
            )  # explicitly enumerate the new run directory
            os.mkdir(self.workDir)
            self.reset()
        else:
            # move to working dir
            self.workDir = self.config.workdir + '/' + 'run%d' %self.config.run_num
            os.chdir(self.workDir)
            printRecord("Resuming run %d" % self.config.run_num)

    def reset(self):
        os.chdir(self.homedir)
        #os.mkdir(f"{self.workDir}/ckpts")
        os.mkdir(f"{self.workDir}/episode{self.episode}")
        os.mkdir(f"{self.workDir}/episode{self.episode}/ckpts")
        os.mkdir(f"{self.workDir}/episode{self.episode}/datasets")
        os.chdir(f"{self.workDir}/episode{self.episode}")  # move to working dir
        printRecord("Starting Fresh Run %d" % self.runNum)
        self.oracle.initializeDataset()  # generate toy model dataset
        self.stateDict = None
        self.sampleDict = None
        self.totalLoss = None
        self.testMinima = None
        self.stateDictRecord = None
        self.reward = None
        self.terminal = None
        self.model = None
        self.cumulativeReward = None
        self.rewardList = None
        self.bottomTenLoss = None
        self.action = None
        self.trueMinimum = None
        self.oracleRecord = None
        self.bestScores = None
        self.prevIterBest = None

    def makeNewWorkingDirectory(self):  # make working directory
        """
        make a new working directory
        non-overlapping previous entries
        :return:
        """
        workdirs = glob.glob(
            self.config.workdir + "/" + "run*"
        )  # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split("run")[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.config.workdir + "/" + "run%d" % (prev_max + 1)
            self.config.workdir = self.workDir
            os.mkdir(self.workDir)
            self.runNum = int(prev_max + 1)
        else:
            self.workDir = self.config.workdir + "/" + "run1"
            os.mkdir(self.workDir)

    def runPipeline(self):
        """
        run  the active learning pipeline for a number of iterations
        :return:
        """
        self.config.dataset_size = self.config.dataset.init_length
        for _ in range(self.config.al.episodes):

            if self.config.dataset.type == "toy":
                self.sampleOracle()  # use the oracle to pre-solve the problem for future benchmarking

            self.testMinima = []  # best test loss of models, for each iteration of the pipeline
            self.bestScores = (
                []
            )  # best optima found by the sampler, for each iteration of the pipeline

            for self.pipeIter in range(self.config.al.n_iter):
                printRecord(
                    f"Starting pipeline iteration #{bcolors.FAIL}%d{bcolors.ENDC}"
                    % int(self.pipeIter + 1)
                )
                if self.pipeIter == (self.config.al.n_iter - 1):
                    self.terminal = 1
                else:
                    self.terminal = 0
                self.iterate()  # run the pipeline
                self.saveOutputs()  # save pipeline outputs

            # Train Policy Network
            self.agent.train(BATCH_SIZE=self.config.al.q_batch_size, dqn_epochs=100, comet = self.comet, iteration=self.episode)
            if self.comet:
                self.comet.log_metric(name='RL Cumulative Reward', value=self.cumulativeReward, step=self.episode)
                self.comet.log_metric(name='RL Cumulative Score', value=self.cumulativeScore, step=self.episode)
            #    self.comet.log_metric(name='RL Agent Training Error', value=self.agent.policy_error, step=self.episode)
                #self.comet.log_curve(name='RL Agent Training Error', x=list(range(100)), y=self.agent.policy_error, step=self.episode)
            if self.config.al.episodes > (self.episode + 1):  # if we are doing multiple al episodes
                self.episode += 1
                self.reset()


            # Save Memory for Agent architecture testing
            # numpy.save(f'{self.workDir}/memory.npy', self.agent.memory.memory)
            # numpy.save(f'{self.workDir}/agent_error.npy', self.agent.policy_error)

            self.agent.save_models()
    def iterate(self):
        """
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        """

        t0 = time.time()
        self.retrainModels(parallel=self.config.proxy.training_parallelism)
        tf = time.time()
        printRecord("Retraining took {} seconds".format(int(tf - t0)))

        t0 = time.time()
        self.getModelState()  # run energy-only sampling and create model state dict
        if self.config.al.hyperparams_learning:  # and (self.pipeIter > 0):
            model_state_prev, model_state_curr = self.agent.updateModelState(
                self.stateDict, self.model
            )
            if model_state_prev is not None:
                self.agent.push_to_buffer(
                    model_state_prev, self.action, model_state_curr, self.reward, self.terminal
                )
            self.action = self.agent.getAction()
        else:
            self.action = None

        query = self.querier.buildQuery(
            self.model, self.stateDict, self.sampleDict, action=self.action, comet=self.comet
        )  # pick Samples to be scored

        if self.comet:
            self.comet.log_metric(name='RL E-U Tradeoff', value=self.querier.config.al.energy_uncertainty_tradeoff, step=self.pipeIter)
            self.comet.log_metric(name='RL Dist Cutoff', value=self.querier.config.al.minima_dist_cutoff, step=self.pipeIter)

        tf = time.time()
        printRecord("Query generation took {} seconds".format(int(tf - t0)))

        t0 = time.time()
        scores = self.oracle.score(query)  # score Samples
        tf = time.time()
        printRecord(
            "Oracle scored"
            + bcolors.OKBLUE
            + " {} ".format(len(scores))
            + bcolors.ENDC
            + "queries with average score of"
            + bcolors.OKGREEN
            + " {:.3f}".format(np.average(scores))
            + bcolors.ENDC
        )
        if not self.config.dataset.type == "toy":
            printRecord("Oracle scoring took {} seconds".format(int(tf - t0)))

        if self.comet:
            self.comet.log_histogram_3d(scores, name="query scores", step=self.pipeIter)

        self.updateDataset(query, scores)  # add scored Samples to dataset

    def getModelState(self):
        """
        sample the model
        report on the status of dataset
        report on best scores according to models
        report on model confidence
        :manualRerun: reruns the sampler even if we already have priorsampler data)
        :return:
        """
        """
        key outputs (not toy):
            - test loss
            - energy and uncertainty of best X distinct samples
        key outputs (toy):
            - large sample loss & bottom x% loss
            - distance to known minimum
            -? number of true minima
        """

        # run the sampler
        self.loadEstimatorEnsemble()
        self.sampleDict = self.querier.runSampling(
            self.model, [1, 0], 1, method_overwrite="mcmc"
        )  # sample existing optima - always construct model state with mcmc
        samples = self.sampleDict["samples"]
        energies = self.sampleDict["energies"]
        uncertainties = self.sampleDict["uncertainties"]

        # agglomerative clustering
        clusters, clusterEns, clusterVars = doAgglomerativeClustering(
            samples,
            energies,
            uncertainties,
            self.config.dataset.dict_size,
            cutoff=normalizeDistCutoff(self.config.al.minima_dist_cutoff),
        )
        (
            clusterSizes,
            avgClusterEns,
            minClusterEns,
            avgClusterVars,
            minClusterVars,
            minClusterSamples,
        ) = clusterAnalysis(clusters, clusterEns, clusterVars)

        # clutering alternative - just include sample-by-sample
        # bestInds = sortTopXSamples(samples[np.argsort(scores)], nSamples=len(samples), distCutoff=0.1)  # sort out the best, and at least minimally distinctive samples

        if (
            len(clusters) < self.config.querier.model_state_size
        ):  # if we don't have enough clusters for the model, pad with random samples from the sampling run
            minClusterSamples, minClusterEns, minClusterVars = self.addRandomSamples(
                samples, energies, uncertainties, minClusterSamples, minClusterEns, minClusterVars
            )

        # get distances to relevant datasets
        internalDist, datasetDist, randomDist = self.getDataDists(
            minClusterSamples[: self.config.querier.model_state_size]
        )
        self.getReward(minClusterEns, minClusterVars)

        self.stateDict = {
            "test loss": np.average(
                self.testMinima
            ),  # losses are evaluated on standardized data, so we do not need to re-standardize here
            "test std": np.sqrt(np.var(self.testMinima)),
            "all test losses": self.testMinima,
            "best cluster energies": (
                minClusterEns[: self.config.querier.model_state_size] - self.model.mean
            )
            / self.model.std,  # standardize according to dataset statistics
            "best cluster deviations": np.sqrt(
                minClusterVars[: self.config.querier.model_state_size]
            )
            / self.model.std,
            "best cluster samples": minClusterSamples[: self.config.querier.model_state_size],
            "best clusters internal diff": internalDist,
            "best clusters dataset diff": datasetDist,
            "best clusters random set diff": randomDist,
            "clustering cutoff": self.config.al.minima_dist_cutoff,  # could be a learned parameter
            "n proxy models": self.config.proxy.ensemble_size,
            "iter": self.pipeIter,
            "budget": self.config.al.n_iter,
            "reward": self.reward,
        }

        printRecord(
            "%d " % self.config.proxy.ensemble_size
            + f"Model ensemble training converged with average test loss of {bcolors.OKCYAN}%.5f{bcolors.ENDC}"
            % np.average(np.asarray(self.testMinima[-self.config.proxy.ensemble_size :]))
            + f" and std of {bcolors.OKCYAN}%.3f{bcolors.ENDC}" % (np.sqrt(np.var(self.testMinima)))
        )
        printRecord(
            "Model state contains {} samples".format(self.config.querier.model_state_size)
            + " with minimum energy"
            + bcolors.OKGREEN
            + " {:.2f},".format(np.amin(minClusterEns))
            + bcolors.ENDC
            + " average energy"
            + bcolors.OKGREEN
            + " {:.2f},".format(np.average(minClusterEns[: self.config.querier.model_state_size]))
            + bcolors.ENDC
            + " and average std dev"
            + bcolors.OKCYAN
            + " {:.2f}".format(
                np.average(np.sqrt(minClusterVars[: self.config.querier.model_state_size]))
            )
            + bcolors.ENDC
        )
        printRecord(
            "Best sample in model state is {}".format(
                numbers2letters(minClusterSamples[np.argmin(minClusterEns)])
            )
        )
        printRecord(
            "Sample average mutual distance is "
            + bcolors.WARNING
            + "{:.2f} ".format(np.average(internalDist))
            + bcolors.ENDC
            + "dataset distance is "
            + bcolors.WARNING
            + "{:.2f} ".format(np.average(datasetDist))
            + bcolors.ENDC
            + "and overall distance estimated at "
            + bcolors.WARNING
            + "{:.2f}".format(np.average(randomDist))
            + bcolors.ENDC
        )

        if (
            self.config.dataset.type == "toy"
        ):  # we can check the test error against a huge random dataset
            self.largeModelEvaluation()
            if self.comet:
                self.comet.log_metric(
                    name="proxy loss on best 10% of large random dataset",
                    value=self.bottomTenLoss[0],
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name="proxy loss on large random dataset",
                    value=self.totalLoss[0],
                    step=self.pipeIter,
                )

        if self.pipeIter == 0:  # if it's the first round, initialize, else, append
            self.stateDictRecord = [self.stateDict]
        else:
            self.stateDictRecord.append(self.stateDict)

        if self.comet:
            self.comet.log_histogram_3d(
                energies, name="model state sampling run energies", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                np.sqrt(uncertainties),
                name="model state sampling run std deviations",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                minClusterEns[: self.config.querier.model_state_size],
                name="model state energies",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                np.sqrt(minClusterVars[: self.config.querier.model_state_size]),
                name="model state std deviations",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                internalDist, name="model state internal distance", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                datasetDist, name="model state distance from dataset", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                randomDist, name="model state distance from large random sample", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                self.testMinima[-1], name="proxy model test minima", step=self.pipeIter
            )

    def getReward(self, bestEns, bestVars):
        """
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        """
        # get the best results in the standardized basis
        best_ens_standardized = (bestEns - self.model.mean) / self.model.std
        standardized_standard_deviations = np.sqrt(bestVars) / self.model.std
        adjusted_standardized_energies = (
            best_ens_standardized + standardized_standard_deviations
        )  # consider std dev as an uncertainty envelope and take the high end
        best_standardized_adjusted_energy = np.amin(adjusted_standardized_energies)

        # convert to raw outputs basis
        adjusted_energies = bestEns + np.sqrt(bestVars)
        best_adjusted_energy = np.amin(adjusted_energies)  # best energy, adjusted for uncertainty
        if self.pipeIter == 0:
            self.reward = 0  # first iteration - can't define a reward
            self.cumulativeReward = 0
            self.rewardList = np.zeros(self.config.al.n_iter)
            self.prevIterBest = [best_adjusted_energy]
        else:  # calculate reward using current standardization
            stdPrevIterBest = (self.prevIterBest[-1] - self.model.mean) / self.model.std
            self.reward = -(
                best_standardized_adjusted_energy - stdPrevIterBest
            )  # reward is the delta between variance-adjusted energies in the standardized basis (smaller is better)
            self.rewardList[self.pipeIter] = self.reward
            self.cumulativeReward = sum(self.rewardList)
            self.prevIterBest.append(best_adjusted_energy)

        printRecord(
            "Iteration best uncertainty-adjusted result = {:.3f}, previous best = {:.3f}, reward = {:.3f}, cumulative reward = {:.3f}".format(
                best_adjusted_energy, self.prevIterBest[-1], self.reward, self.cumulativeReward
            )
        )

        if (
            self.config.dataset.type == "toy"
        ):  # if it's  a toy dataset, report the cumulative performance against the known minimum
            stdTrueMinimum = (self.trueMinimum - self.model.mean) / self.model.std
            if self.pipeIter == 0:
                self.abs_score = [
                    1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)
                ]
                self.cumulativeScore = 0
            elif self.pipeIter > 0:
                # we will compute the distance from our best answer to the correct answer and integrate it over the number of samples in the dataset
                xaxis = (
                    self.config.dataset_size
                    + np.arange(0, self.pipeIter + 1) * self.config.al.queries_per_iter
                )  # how many samples in the dataset used for each
                self.abs_score.append(
                    1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)
                )  # compute proximity to correct answer in standardized basis
                self.cumulativeScore = np.trapz(self.abs_score, x=xaxis)
                self.normedCumScore = self.cumulativeScore / xaxis[-1]
                printRecord(
                    "Total score is {:.3f} and {:.5f} per-sample after {} samples".format(
                        self.abs_score[-1], self.normedCumScore, xaxis[-1]
                    )
                )
            else:
                print("Error! Pipeline iteration cannot be negative")
                sys.exit()

            if self.comet:
                self.comet.log_metric(
                    name="absolute score", value=self.abs_score[-1], step=self.pipeIter
                )
                self.comet.log_metric(
                    name="cumulative score", value=self.cumulativeScore, step=self.pipeIter
                )
                self.comet.log_metric(name="reward", value=self.reward, step=self.pipeIter)

    def retrainModels(self, parallel=True):
        if not parallel:
            testMins = []
            for i in range(self.config.proxy.ensemble_size):
                self.resetModel(
                    i
                )  # reset between ensemble estimators EVERY ITERATION of the pipeline
                self.model.converge()  # converge model
                testMins.append(np.amin(self.model.err_te_hist))
                if self.comet:
                    tr_hist = self.model.err_tr_hist
                    te_hist = self.model.err_te_hist
                    epochs = len(te_hist)
                    for i in range(epochs):
                        self.comet.log_metric(
                            "proxy train loss iter {}".format(self.pipeIter),
                            step=i,
                            value=tr_hist[i],
                        )
                        self.comet.log_metric(
                            "proxy test loss iter {}".format(self.pipeIter),
                            step=i,
                            value=te_hist[i],
                        )

            self.testMinima.append(testMins)
        else:  # deprecated
            del self.model
            if self.config.machine == "local":
                nHold = 4
            else:
                nHold = 1
            cpus = int(os.cpu_count() - nHold)
            cpus = min(cpus, self.config.proxy.ensemble_size)  # only as many CPUs as we need
            with mp.Pool(processes=cpus) as pool:
                output = [
                    pool.apply_async(trainModel, args=[self.config, j])
                    for j in range(self.config.proxy.ensemble_size)
                ]
                outputList = [output[i].get() for i in range(self.config.proxy.ensemble_size)]
                self.testMinima.append(
                    [np.amin(outputList[i]) for i in range(self.config.proxy.ensemble_size)]
                )
                pool.close()
                pool.join()

    def loadEstimatorEnsemble(self):
        """
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        """
        ensemble = []
        for i in range(self.config.proxy.ensemble_size):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = modelNet(self.config, 0)
        self.model.loadEnsemble(ensemble)

        # print('Loaded {} estimators'.format(int(self.config.proxy.ensemble_size)))

    def resetModel(self, ensembleIndex, returnModel=False):
        """
        load a new instance of the model with reset parameters
        :return:
        """
        try:  # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = modelNet(self.config, ensembleIndex)
        # printRecord(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))
        if returnModel:
            return self.model

    def getModelSize(self):
        self.model = modelNet(self.config, 0)
        nParams = get_n_params(self.model.model)
        printRecord("Proxy model has {} parameters".format(int(nParams)))
        del self.model

    def largeModelEvaluation(self):
        """
        if we are using a toy oracle, we should be able to easily get the test loss on a huge sample of the dataset
        :return:
        """
        self.loadEstimatorEnsemble()

        numSamples = min(
            int(1e4), self.config.dataset.dict_size ** self.config.dataset.max_length // 100
        )  # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(
            save=False, returnData=True, customSize=numSamples
        )  # get large random dataset
        randomSamples = randomData["samples"]
        randomScores = randomData["scores"]

        sortInds = np.argsort(randomScores)  # sort randoms
        randomSamples = randomSamples[sortInds]
        randomScores = randomScores[sortInds]

        modelScores, modelVars = [[], []]
        sampleLoader = data.DataLoader(
            randomSamples,
            batch_size=self.config.proxy.mbsize,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        for i, testData in enumerate(sampleLoader):
            score, variance = self.model.evaluate(testData.float(), output="Both")
            modelScores.extend(score)
            modelVars.extend(variance)

        bestTenInd = numSamples // 10
        totalLoss = F.smooth_l1_loss(
            (torch.Tensor(modelScores).float() - self.model.mean) / self.model.std,
            (torch.Tensor(randomScores).float() - self.model.mean) / self.model.std,
        )  # full dataset loss (standardized basis)
        bottomTenLoss = F.smooth_l1_loss(
            (torch.Tensor(modelScores[:bestTenInd]).float() - self.model.mean) / self.model.std,
            (torch.Tensor(randomScores[:bestTenInd]).float() - self.model.mean) / self.model.std,
        )  # bottom 10% loss (standardized basis)

        if self.pipeIter == 0:  # if it's the first round, initialize, else, append
            self.totalLoss = [totalLoss]
            self.bottomTenLoss = [bottomTenLoss]
        else:
            self.totalLoss.append(totalLoss)
            self.bottomTenLoss.append(bottomTenLoss)

        printRecord(
            "Model has overall loss of"
            + bcolors.OKCYAN
            + " {:.5f}, ".format(totalLoss)
            + bcolors.ENDC
            + "best 10% loss of"
            + bcolors.OKCYAN
            + " {:.5f} ".format(bottomTenLoss)
            + bcolors.ENDC
            + "on {} toy dataset samples".format(numSamples)
        )

    def sampleOracle(self):
        """
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        """
        printRecord("Asking toy oracle for the true minimum")

        self.model = "abc"
        gammas = np.logspace(
            self.config.mcmc.stun_min_gamma,
            self.config.mcmc.stun_max_gamma,
            self.config.mcmc.num_samplers,
        )
        mcmcSampler = Sampler(self.config, 0, [1, 0], gammas)
        if (
            (self.config.dataset.oracle == "linear")
            or (self.config.dataset.oracle == "nupack energy")
            or (self.config.dataset.oracle == "5 pairs")
        ):
            samples = mcmcSampler.sample(
                self.model, useOracle=True, nIters=100
            )  # do a tiny number of iters - the minimum is known
        else:
            samples = mcmcSampler.sample(self.model, useOracle=True)  # do a genuine search

        sampleDict = samples2dict(samples)
        if (
            self.config.dataset.oracle == "wmodel"
        ):  # w model minimum is always zero - even if we don't find it
            bestMin = 0
        else:
            bestMin = np.amin(sampleDict["energies"])

        # append suggestions for known likely solutions
        if self.config.dataset.oracle == "linear":
            goodSamples = np.zeros(
                (4, self.config.dataset.max_length)
            )  # all of one class usually best
            goodSamples[0] = goodSamples[1] + 1
            goodSamples[1] = goodSamples[1] + 2
            goodSamples[2] = goodSamples[2] + 3
            goodSamples[3] = goodSamples[3] + 4
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                printRecord("Pre-loaded minimum was better than one found by sampler")

        elif (self.config.dataset.oracle == "nupack energy") or (
            self.config.dataset.oracle == "nupack pairs"
        ):
            goodSamples = (
                np.ones((4, self.config.dataset.max_length)) * 4
            )  # GCGC CGCG GGGCCC CCCGGG
            goodSamples[0, 0:-1:2] = 3
            goodSamples[1, 1:-1:2] = 3
            goodSamples[2, : self.config.dataset.max_length // 2] = 3
            goodSamples[3, self.config.dataset.max_length // 2 :] = 3
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                printRecord("Pre-loaded minimum was better than one found by sampler")

        printRecord(
            f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % bestMin
            + " from %d" % self.config.mcmc.num_samplers
            + " sampling runs."
        )
        printRecord(
            "Best sample found is {}".format(
                numbers2letters(sampleDict["samples"][np.argmin(sampleDict["energies"])])
            )
        )

        self.oracleRecord = sampleDict
        self.trueMinimum = bestMin

        if self.comet:
            self.comet.log_histogram_3d(sampleDict["energies"], name="energies_true", step=0)

    def saveOutputs(self):
        """
        save config and outputs in a dict
        :return:
        """
        outputDict = {}
        outputDict["config"] = Namespace(**dict(vars(self.config)))
        if "comet" in outputDict["config"]:
            del outputDict["config"].comet
        outputDict["state dict record"] = self.stateDictRecord
        outputDict["rewards"] = self.rewardList
        if self.config.dataset.type == "toy":
            outputDict["oracle outputs"] = self.oracleRecord
            outputDict["big dataset loss"] = self.totalLoss
            outputDict["bottom 10% loss"] = self.bottomTenLoss
            if self.pipeIter > 1:
                outputDict["score record"] = self.abs_score
                outputDict["cumulative score"] = (self.cumulativeScore,)
                outputDict["per sample cumulative score"] = self.normedCumScore
        np.save("outputsDict", outputDict)

    def updateDataset(self, oracleSequences, oracleScores):
        """
        loads dataset, appends new datapoints from oracle, and saves dataset
        :param params: model parameters
        :param oracleSequences: sequences which were sent to oracle
        :param oracleScores: scores of sequences sent to oracle
        :return: n/a
        """
        dataset = np.load(
            "datasets/" + self.config.dataset.oracle + ".npy", allow_pickle=True
        ).item()
        # TODO separate between scores and q-scores
        dataset["samples"] = np.concatenate((dataset["samples"], oracleSequences))
        dataset["scores"] = np.concatenate((dataset["scores"], oracleScores))

        if self.comet:
            self.comet.log_histogram_3d(
                dataset["scores"], name="dataset scores", step=self.pipeIter
            )
            idx_sorted = np.argsort(dataset["scores"])
            for k in [1, 10, 100]:
                topk_scores = dataset["scores"][idx_sorted[:k]]
                topk_samples = dataset["samples"][idx_sorted[:k]]
                dist = binaryDistance(topk_samples, pairwise=False, extractInds=len(topk_samples))
                self.comet.log_metric(
                    f"mean top-{k} scores", np.mean(topk_scores), step=self.pipeIter
                )
                self.comet.log_metric(
                    f"std top-{k} scores", np.std(topk_scores), step=self.pipeIter
                )
                self.comet.log_metric(f"mean dist top-{k}", np.mean(dist), step=self.pipeIter)

        self.config.dataset_size = len(dataset["samples"])

        printRecord(
            f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" % int(len(oracleSequences))
            + " to the dataset, total dataset size is"
            + bcolors.OKBLUE
            + " {}".format(int(len(dataset["samples"])))
            + bcolors.ENDC
        )
        printRecord(
            bcolors.UNDERLINE
            + "====================================================================="
            + bcolors.ENDC
        )
        np.save("datasets/" + self.config.dataset.oracle, dataset)
        np.save(
            "datasets/" + self.config.dataset.oracle + "_iter_{}".format(self.pipeIter), dataset
        )

        if self.comet:
            self.comet.log_histogram_3d(
                dataset["scores"], name="dataset scores", step=self.pipeIter
            )
            dataset2 = dataset.copy()
            dataset2["samples"] = numbers2letters(dataset["samples"])
            self.comet.log_table(
                filename="dataset_at_iter_{}.csv".format(self.pipeIter),
                tabular_data=pd.DataFrame.from_dict(dataset2),
            )

    def getScalingFactor(self):
        """
        since regression is not normalized, we identify a scaling factor against which we normalize our results
        :return:
        """
        truncationFactor = 0.1  # cut off x% of the furthest outliers
        dataset = np.load(
            "datasets/" + self.config.dataset.oracle + ".npy", allow_pickle=True
        ).item()

        scores = dataset["scores"]
        d1 = [np.sum(np.abs(scores[i] - scores)) for i in range(len(scores))]
        scores = scores[np.argsort(d1)]  # sort according to mutual distance
        margin = int(len(scores) * truncationFactor)
        scores = scores[:-margin]  # cut 'margin' of furthest points
        self.scalingFactor = np.ptp(scores)

    def addRandomSamples(
        self, samples, energies, uncertainties, minClusterSamples, minClusterEns, minClusterVars
    ):
        rands = np.random.randint(
            0, len(samples), size=self.config.querier.model_state_size - len(minClusterSamples)
        )
        randomSamples = samples[rands]
        randomEnergies = energies[rands]
        randomUncertainties = uncertainties[rands]
        minClusterSamples = np.concatenate((minClusterSamples, randomSamples))
        minClusterEns = np.concatenate((minClusterEns, randomEnergies))
        minClusterVars = np.concatenate((minClusterVars, randomUncertainties))
        printRecord("Padded model state with {} random samples from sampler run".format(len(rands)))

        return minClusterSamples, minClusterEns, minClusterVars

    def getDataDists(self, samples):
        """
        compute average binary distances between a set of samples and
        1 - itself
        2 - the training dataset
        3 - a large random sample
        :param samples:
        :return:
        """
        # training dataset
        dataset = np.load(
            "datasets/" + self.config.dataset.oracle + ".npy", allow_pickle=True
        ).item()
        dataset = dataset["samples"]

        # large, random sample
        numSamples = min(
            int(1e4), self.config.dataset.dict_size ** self.config.dataset.max_length // 100
        )  # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(
            save=False, returnData=True, customSize=numSamples
        )  # get large random dataset
        randomSamples = randomData["samples"]

        internalDist = binaryDistance(
            samples, self.config.dataset.dict_size, pairwise=False, extractInds=len(samples)
        )
        datasetDist = binaryDistance(
            np.concatenate((samples, dataset)),
            self.config.dataset.dict_size,
            pairwise=False,
            extractInds=len(samples),
        )
        randomDist = binaryDistance(
            np.concatenate((samples, randomSamples)),
            self.config.dataset.dict_size,
            pairwise=False,
            extractInds=len(samples),
        )

        return internalDist, datasetDist, randomDist


def trainModel(config, i):
    """
    rewritten for training in a parallelized fashion
    needs to be outside the class method for multiprocessing to work
    :param i:
    :return:
    """

    model = modelNet(config, i)
    err_te_hist = model.converge(returnHist=True)  # converge model

    return err_te_hist

