"""The main entry point for performing comparison on chatbots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
sys.path.append('./')


# import cohere
# import openai
# import anthropic
import pandas as pd

from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.models import global_models
from zeno_build.optimizers import standard
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn
from zeno_build.reporting.visualize import visualize

import priv_key
import config as chatbot_config
from modeling import make_predictions


def chatbot_main(
    results_dir: str,
    cached_data: str | None = None,
    cached_results: str | None = None,
    do_visualization: bool = True,
):
    """Run the chatbot experiment."""
    os.environ['INSPIREDCO_API_KEY'] = priv_key.inspiredco_api_key

    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the necessary data
    print('Loading data...')
    with open(cached_data, "r", encoding='utf-8') as f:
        contexts_and_labels = [
            ChatMessages(
                messages=[
                    ChatTurn(role=y["role"], content=y["content"])
                    for y in x["messages"]
                ]
            )
            for x in json.load(f)
        ]
    print(f"Loaded test date: {len(contexts_and_labels)} items")

    # Organize the data into source and context
    labels: list[str] = []
    contexts: list[ChatMessages] = []
    for x in contexts_and_labels:
        labels.append(x.messages[-1].content)
        contexts.append(ChatMessages(x.messages[:-1]))

    # Run the hyperparameter sweep and print out results
    results: list[ExperimentRun] = []
    if cached_results is not None:
        print("Loading cached result...")
        with open(cached_results, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
        print(f"Loaded cached result: {len(results)} items (prediction_model)")
    else:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=chatbot_config.space,
            constants=chatbot_config.constants,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
        )
        for _ in range(chatbot_config.num_trials):
            parameters = optimizer.get_parameters()
            print(f"{_}th prediction, the parameters are: {parameters}")

            predictions = make_predictions(
                data=contexts,
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                cache_root=os.path.join(results_dir, "cache"),
            )
            eval_result = optimizer.calculate_metric(contexts, labels, predictions)

            run = ExperimentRun(
                parameters=parameters,
                predictions=predictions,
                eval_result=eval_result,
            )
            results.append(run)

        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "cached_results.json"), "w") as f:
            json.dump(serialized_results, f)


    for run in results:
        if run.name is None:
            run.name = " ".join(
                [
                    run.parameters[k]
                    if isinstance(run.parameters[k], str)
                    else f"{k}={run.parameters[k]}"
                    for k in chatbot_config.space.keys()
                ]
            )

    # Perform the visualization
    if do_visualization:
        df = pd.DataFrame(
            {
                "messages": [[asdict(y) for y in x.messages] for x in contexts],
                "label": labels,
            }
        )
        visualize(
            df,
            labels,
            results,
            "openai-chat",
            "messages",
            chatbot_config.zeno_distill_and_metric_functions,
        )


if __name__ == "__main__":
    # python zeno_eval.py --cached_data eval_data/eval_turn_data_cached.json --skip_visualization

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ZenoEval/zeno_results",
        help="The directory to store the results in.",
    )
    parser.add_argument(
        "--cached_data",
        type=str,
        default=None,
        help="A path to a json file with the cached data.",
    )
    parser.add_argument(
        "--cached_results",
        type=str,
        default=None,
        help="A path to a json file with cached runs.",
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Whether to skip the visualization step.",
    )
    args = parser.parse_args()

    chatbot_main(
        results_dir=args.results_dir,
        cached_data=args.cached_data,
        cached_results=args.cached_results,
        do_visualization=not args.skip_visualization,
    )
