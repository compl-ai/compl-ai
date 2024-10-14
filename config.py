import os
from pathlib import Path

# Ignore flake8 since we have imports all over the place
# flake8: noqa
# isort: skip_file
from helper_tools.results_processor import (
    reformat_bbq_metric,
    reformat_bold_metric,
    reformat_calibration_metric,
    reformat_consistency_metric,
    reformat_decoding_trust_metric,
    reformat_fairllm_metric,
    reformat_human_deception,
    reformat_human_eval_metric,
    reformat_instruction_goal_hijacking_metric,
    reformat_mcq_metric,
    reformat_memorization_metric,
    reformat_mmlu_robustness,
    reformat_multiturn_goal_hijacking_metric,
    reformat_privacy_metric,
    reformat_reddit_bias_metric,
    reformat_toxicity_metric,
    reformat_toxicity_advbench_metric,
    reformat_watermarking_metric,
    reformat_training_data_suitability,
)
from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.base_data import BaseData
from src.metrics.base_metric import BaseMetric
from src.registry import ComponentRegistry
from src.registry import registry
from src.registry import BENCHMARK_PROCESSORS

from src.benchmarks import multiple_choice_benchmark
from src.benchmarks.benchmark_implementations.capabilities import (
    ai2_reasoning,
    hellaswag,
    human_eval,
    mmlu,
)
from src.benchmarks.benchmark_implementations.capabilities.truthful_qa import (
    truthful_qa_generation,
    truthful_qa_mc,
)

# Import all the benchmark implementations
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.configs.base_metric_config import MetricConfig
from src.data.hf_data import HFDataConfig

from src.metrics import hf_metric
from src.metrics.metric_scripts import truthful_scores, normalized_total_prob

####
# General global constants
####

# Directory from where to include other configs
CONFIG_DIR = os.path.abspath("configs/")
RESULTS_DIR = os.path.abspath("results/")
CODE_ROOT_PATH = Path(os.path.abspath(__file__)) / "src"

metric_registry = ComponentRegistry(BaseMetric, MetricConfig)
dataset_registry = ComponentRegistry(BaseData, DataConfig)
benchmark_registry = ComponentRegistry(BaseBenchmark, BenchmarkConfig)

registry.register("metric", metric_registry)
registry.register("data", dataset_registry)
registry.register("benchmark", benchmark_registry)

####
# Capabilities
####


# mmlu and ai2reasoning & hellaswag have a benchmark implementation
benchmark_registry.register_logic_config_classes(
    "mmlu", mmlu.MMLU, mmlu.MMLUConfig, category="capabilities"
)
benchmark_registry.register_logic_config_classes(
    "multiple_choice",
    multiple_choice_benchmark.MultipleChoice,
    multiple_choice_benchmark.MultipleChoiceConfig,
    category="capabilities",
)

benchmark_registry.register_logic_config_classes(
    "truthful_qa_generation",
    truthful_qa_generation.TruthfulQAGeneration,
    multiple_choice_benchmark.MultipleChoiceConfig,
    category="capabilities",
)

benchmark_registry.register_logic_config_classes(
    "hellaswag", hellaswag.Hellaswag, BenchmarkConfig, category="capabilities"
)


benchmark_registry.register_logic_config_classes(
    "human_eval", human_eval.HumanEval, BenchmarkConfig, category="capabilities"
)

dataset_registry.register_logic_config_classes("mmlu_data", mmlu.MMLUData, HFDataConfig)


dataset_registry.register_logic_config_classes(
    "ai2_reasoning", ai2_reasoning.AI2ReasoningData, HFDataConfig
)

benchmark_registry.register_logic_config_classes(
    "ai2_reasoning",
    ai2_reasoning.AI2Reasoning,
    BenchmarkConfig,
    category="capabilities",
)

dataset_registry.register_logic_config_classes("hellaswag", hellaswag.HellaswagData, HFDataConfig)

dataset_registry.register_logic_config_classes(
    "truthful_qa",
    truthful_qa_mc.TruthfulQAMCData,
    truthful_qa_mc.TruthfulQAMCDataConfig,
)

benchmark_registry.register_logic_config_classes(
    "truthful_qa_mc2",
    truthful_qa_mc.TruthfulQAMC2,
    BenchmarkConfig,
    category="capabilities",
)

dataset_registry.register_logic_config_classes(
    "truthful_qa_generation",
    truthful_qa_generation.TruthfulQAGenerationData,
    HFDataConfig,
)

metric_registry.register_logic_config_classes("hf_metric", hf_metric.HFMetric, MetricConfig)
metric_registry.register_logic_config_classes(
    "truthful_scores",
    truthful_scores.TruthfulScores,
    MetricConfig,
)
metric_registry.register_logic_config_classes(
    "normalized_total_prob", normalized_total_prob.NormalizedTotalProbabilities, MetricConfig
)

####
# Robustness
####

from src.benchmarks.benchmark_implementations.robustness import (
    boolq_contrast,
    imdb_contrast,
)

dataset_registry.register_logic_config_classes(
    "boolq_contrast",
    boolq_contrast.BoolQContrastData,
    HFDataConfig,
    category="robustness",
)

dataset_registry.register_logic_config_classes(
    "imdb_contrast",
    imdb_contrast.IMDBContrastData,
    HFDataConfig,
)

from src.benchmarks.benchmark_implementations.consistency import (
    forecast_consistency,
    self_check_consistency,
)


####
# Calibration
####

from src.benchmarks.benchmark_implementations.calibration import (
    calibration_mc_big_bench,
    calibration_mc_i_know,
)

benchmark_registry.register_logic_config_classes(
    "calibration_mc_big_bench",
    calibration_mc_big_bench.TrueFalseCalibration,
    calibration_mc_big_bench.TrueFalseCalibrationConfig,
    category="calibration",
)

dataset_registry.register_logic_config_classes(
    "calibration_mc_big_bench",
    calibration_mc_big_bench.TrueFalseCalibrationData,
    HFDataConfig,
)

benchmark_registry.register_logic_config_classes(
    "i_know_calibration",
    calibration_mc_i_know.IKnowCalibration,
    calibration_mc_big_bench.TrueFalseCalibrationConfig,
    category="calibration",
)

dataset_registry.register_logic_config_classes(
    "i_know_calibration",
    calibration_mc_i_know.IKnowCalibrationData,
    HFDataConfig,
)


####
# Consistency
####

benchmark_registry.register_logic_config_classes(
    "forecast_consistency",
    forecast_consistency.ForecastConsistency,
    forecast_consistency.ForecastConsistencyConfig,
    category="consistency",
)

dataset_registry.register_logic_config_classes(
    "forecast_consistency",
    forecast_consistency.ForecastData,
    forecast_consistency.ForecastDataConfig,
)

benchmark_registry.register_logic_config_classes(
    "self_check_consistency",
    self_check_consistency.SelfCheckConsistency,
    self_check_consistency.SelfCheckConsistencyConfig,
    category="consistency",
)

dataset_registry.register_logic_config_classes(
    "self_check_consistency",
    self_check_consistency.SelfCheckConsistencyData,
    self_check_consistency.SelfCheckConsistencyDataConfig,
)


####
# Cybersecurity
####

from src.benchmarks.benchmark_implementations.cybersecurity import (
    instruction_goal_hijacking,
)
from src.data.tensortrust_data import TensortrustData, TensortrustDataConfig

benchmark_registry.register_logic_config_classes(
    "instruction_goal_hijacking",
    instruction_goal_hijacking.InstructionGoalHijacking,
    instruction_goal_hijacking.InstructionGoalHijackingConfig,
    category="cybersecurity",
)

dataset_registry.register_logic_config_classes(
    "instruction_goal_hijacking", TensortrustData, TensortrustDataConfig
)

from src.benchmarks.benchmark_implementations.cybersecurity import (
    dynamic_goal_hijacking,
    multiturn_goal_hijacking,
)

benchmark_registry.register_logic_config_classes(
    "multiturn_goal_hijacking",
    multiturn_goal_hijacking.LanguageGame,
    multiturn_goal_hijacking.LanguageGameConfig,
    category="cybersecurity",
)

dataset_registry.register_logic_config_classes(
    "multiturn_goal_hijacking",
    multiturn_goal_hijacking.LanguageGameData,
    multiturn_goal_hijacking.LanguageGameDataConfig,
)

benchmark_registry.register_logic_config_classes(
    "dynamic_goal_hijacking",
    dynamic_goal_hijacking.DynamicGoalHijacking,
    dynamic_goal_hijacking.DynamicGoalHijackingConfig,
    category="cybersecurity",
)


####
# Bias
####

from src.benchmarks.benchmark_implementations.bias import reddit_bias

benchmark_registry.register_logic_config_classes(
    "reddit_bias", reddit_bias.RedditBias, reddit_bias.RedditBiasConfig, category="bias"
)

dataset_registry.register_logic_config_classes(
    "reddit_bias_data",
    reddit_bias.RedditDataProvider,
    reddit_bias.RedditDataConfig,
)


####
# Fairness
####

from src.benchmarks.benchmark_implementations.fairness import decoding_trust

benchmark_registry.register_logic_config_classes(
    "decoding_trust",
    decoding_trust.DecodingTrustBenchmark,
    decoding_trust.DecodingTrustConfig,
    category="fairness",
)

dataset_registry.register_logic_config_classes(
    "decoding_trust",
    decoding_trust.DecodingTrustDataProvider,
    decoding_trust.DecodingTrustDataConfig,
)

from src.benchmarks.benchmark_implementations.fairness.bbq import bbq_benchmark

benchmark_registry.register_logic_config_classes(
    "bbq",
    bbq_benchmark.BBQ,
    bbq_benchmark.BBQConfig,
    category="fairness",
)

dataset_registry.register_logic_config_classes(
    "bbq",
    bbq_benchmark.BBQData,
    bbq_benchmark.BBQDataConfig,
)

from src.benchmarks.benchmark_implementations.fairness.bold import bold_benchmark

benchmark_registry.register_logic_config_classes(
    "bold",
    bold_benchmark.Bold,
    bold_benchmark.BoldConfig,
    category="fairness",
)

dataset_registry.register_logic_config_classes(
    "bold",
    bold_benchmark.BoldData,
    bold_benchmark.BoldDataConfig,
)

from src.benchmarks.benchmark_implementations.fairness import fairllm

benchmark_registry.register_logic_config_classes(
    "fairllm",
    fairllm.FairLLM,
    BenchmarkConfig,
    category="fairness",
)


####
# Memorization
####


from src.benchmarks.benchmark_implementations.memorization import copyright

benchmark_registry.register_logic_config_classes(
    "memorization",
    copyright.Memorization,
    BenchmarkConfig,
    category="memorization",
)

dataset_registry.register_logic_config_classes(
    "pile_sequences",
    copyright.PileSequences,
    DataConfig,
)

dataset_registry.register_logic_config_classes(
    "books_sequences",
    copyright.BooksSequencesHF,
    DataConfig,
)

####
# Privacy
####


from src.benchmarks.benchmark_implementations.privacy_protection import pii_leak

benchmark_registry.register_logic_config_classes(
    "pii_leak",
    pii_leak.PIILeak,
    BenchmarkConfig,
    category="privacy",
)

dataset_registry.register_logic_config_classes(
    "pii_data",
    pii_leak.PIIData,
    DataConfig,
)
####
# Training Data Suitability
####

from src.benchmarks.benchmark_implementations.training_data_suitability import (
    training_data_suitability,
)

benchmark_registry.register_logic_config_classes(
    "training_data_suitability",
    training_data_suitability.SuitabilityBenchmark,
    training_data_suitability.SuitabilityConfig,
    category="training_data_suitability",
)

dataset_registry.register_logic_config_classes(
    "training_data_suitability", training_data_suitability.SuitabilityData, HFDataConfig
)

####
# Human deception
####

from src.benchmarks.benchmark_implementations.human_deception import human_deception

benchmark_registry.register_logic_config_classes(
    "human_deception",
    human_deception.HumanDeceptionBenchmark,
    BenchmarkConfig,
    category="transparency",
)


dataset_registry.register_logic_config_classes(
    "human_deception",
    human_deception.HumanDeceptionData,
    DataConfig,
)

####
# Traceability
####

from src.benchmarks.benchmark_implementations.traceability import watermarking

benchmark_registry.register_logic_config_classes(
    "watermarking",
    watermarking.WatermarkingBenchmark,
    BenchmarkConfig,
    category="traceability",
)

dataset_registry.register_logic_config_classes(
    "ghostbuster_reddit_wp",
    watermarking.GhostbusterRedditWPData,
    DataConfig,
)


####
# Toxicity
####

from src.benchmarks.benchmark_implementations.toxicity import toxicity

benchmark_registry.register_logic_config_classes(
    "toxicity",
    toxicity.ToxicityBenchmark,
    toxicity.ToxicityConfig,
    category="toxicity",
)

dataset_registry.register_logic_config_classes(
    "toxicity",
    toxicity.ToxicityData,
    toxicity.ToxicityDataConfig,
)

from src.benchmarks.benchmark_implementations.toxicity import toxicity_advbench

benchmark_registry.register_logic_config_classes(
    "toxicity_advbench",
    toxicity_advbench.ToxicityAdvBenchBenchmark,
    toxicity_advbench.ToxicityAdvBenchConfig,
    category="toxicity",
)

dataset_registry.register_logic_config_classes(
    "toxicity_advbench",
    toxicity_advbench.ToxicityAdvBenchData,
    toxicity_advbench.ToxicityAdvBenchDataConfig,
)


####
# Benchmark Postprocessing
####
BENCHMARK_PROCESSORS |= {
    "bbq": reformat_bbq_metric,
    "bold": reformat_bold_metric,
    "toxicity": reformat_toxicity_metric,
    "toxicity_advbench": reformat_toxicity_advbench_metric,
    "forecasting_consistency": reformat_consistency_metric,
    "self_check_consistency": reformat_consistency_metric,
    "boolq_contrast_robustness": reformat_mcq_metric,
    "imdb_contrast_robustness": reformat_mcq_metric,
    "calibration_big_bench": reformat_calibration_metric,
    "calibration_big_bench_i_know": reformat_calibration_metric,
    "decoding_trust": reformat_decoding_trust_metric,
    "hellaswag": reformat_mcq_metric,
    "human_eval": reformat_human_eval_metric,
    "instruction_goal_hijacking": reformat_instruction_goal_hijacking_metric,
    "multiturn_goal_hijacking": reformat_multiturn_goal_hijacking_metric,
    "reddit_bias": reformat_reddit_bias_metric,
    "truthful_qa_mc2": reformat_mcq_metric,
    "mmlu": reformat_mcq_metric,
    "ai2_reasoning": reformat_mcq_metric,
    "human_deception": reformat_human_deception,
    "memorization": reformat_memorization_metric,
    "privacy": reformat_privacy_metric,
    "fairllm": reformat_fairllm_metric,
    "mmlu_robustness": reformat_mmlu_robustness,
    "training_data_suitability": reformat_training_data_suitability,
    "watermarking": reformat_watermarking_metric,
}
