import pandas as pd
import os
import sys
from loguru import logger
from argparse import ArgumentParser
from tqdm import tqdm  # For progress bar

# Add project root to sys.path to allow imports from sofagent
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sofagent.systems import CollaborationSystem
from sofagent.utils import init_api, read_json

# --- Configuration ---
DEFAULT_QUESTIONS_CSV_PATH = "data/evaluation/questions.csv"
DEFAULT_API_CONFIG_PATH = "config/api-config.json"
DEFAULT_SYSTEM_CONFIG_PATH = "config/systems/collaboration/chat.json"
QUESTIONS_COLUMN_NAME = "questions"  # Column in CSV containing the prompts
RESPONSE_COLUMN_NAME = "response"  # Column in CSV to store the responses

# --- Logger Setup ---
logger.remove()
logger.add(sys.stderr, level="INFO")  # Console logging
log_file_path = os.path.join(project_root, "logs", "evaluation_log_{time:YYYY-MM-DD_HH-mm-ss}.log")
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
logger.add(log_file_path, level="DEBUG")  # File logging


def run_evaluation(questions_file: str, api_config_file: str, system_config_file: str):
    """
    Runs the evaluation by processing questions from a CSV file.
    """
    logger.info(f"Starting evaluation with questions from: {questions_file}")

    # --- 1. Load API Configuration and Initialize API ---
    try:
        api_config = read_json(api_config_file)
        init_api(api_config)
        logger.info("API initialized successfully.")
    except FileNotFoundError:
        logger.error(f"API configuration file not found: {api_config_file}")
        return
    except Exception as e:
        logger.error(f"Error initializing API: {e}")
        return

    # --- 2. Instantiate Collaboration System ---
    try:
        # web_demo=False because this is a batch script, not an interactive UI
        system = CollaborationSystem(config_path=system_config_file, task='chat', web_demo=False)
        logger.info("CollaborationSystem instantiated successfully.")
    except FileNotFoundError:
        logger.error(f"System configuration file not found: {system_config_file}")
        return
    except Exception as e:
        logger.error(f"Error instantiating CollaborationSystem: {e}")
        return

    # --- 3. Load Questions CSV ---
    try:
        df = pd.read_csv(questions_file)
        logger.info(f"Loaded {len(df)} questions from '{questions_file}'.")
    except FileNotFoundError:
        logger.error(f"Questions CSV file not found: {questions_file}")
        return
    except Exception as e:
        logger.error(f"Error reading questions CSV: {e}")
        return

    if QUESTIONS_COLUMN_NAME not in df.columns:
        logger.error(f"CSV file must contain a column named '{QUESTIONS_COLUMN_NAME}'.")
        return

    # Add 'response' column if it doesn't exist
    if RESPONSE_COLUMN_NAME not in df.columns:
        df[RESPONSE_COLUMN_NAME] = pd.NA  # Or use None or ""

    # --- 4. Process Each Question and Store Response ---
    logger.info("Processing questions...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Questions"):
        question = row[QUESTIONS_COLUMN_NAME]

        if pd.isna(question) or not str(question).strip():
            logger.warning(f"Skipping empty question at row {index + 2}.")
            df.loc[index, RESPONSE_COLUMN_NAME] = "SKIPPED - Empty Question"
            continue

        logger.debug(f"Processing question (Row {index + 2}): '{question}'")
        try:
            # Use the new method in CollaborationSystem
            response = system.get_response_for_prompt(str(question))
            df.loc[index, RESPONSE_COLUMN_NAME] = response
            logger.debug(f"Response for row {index + 2}: '{response}'")
        except Exception as e:
            logger.error(f"Error processing question at row {index + 2} ('{question}'): {e}", exc_info=True)
            df.loc[index, RESPONSE_COLUMN_NAME] = f"ERROR: {e}"

        # Save periodically to avoid data loss on long runs
        try:
            df.to_csv(questions_file, index=False, encoding='utf-8-sig')
            logger.info(f"Saved intermediate results to '{questions_file}' at row {index + 2}.")
        except Exception as e:
            logger.error(f"Error saving intermediate CSV: {e}")

    # --- 5. Save Updated CSV ---
    try:
        df.to_csv(questions_file, index=False,
                  encoding='utf-8-sig')  # utf-8-sig for Excel compatibility with special chars
        logger.success(f"Evaluation complete. Responses saved to '{questions_file}'.")
    except Exception as e:
        logger.error(f"Error saving final CSV: {e}")


def main():
    parser = ArgumentParser(description="Evaluate SofAgent with questions from a CSV file.")
    parser.add_argument(
        "--questions_csv",
        type=str,
        default=DEFAULT_QUESTIONS_CSV_PATH,
        help=f"Path to the CSV file containing questions (default: {DEFAULT_QUESTIONS_CSV_PATH})"
    )
    parser.add_argument(
        "--api_config",
        type=str,
        default=DEFAULT_API_CONFIG_PATH,
        help=f"Path to the API configuration file (default: {DEFAULT_API_CONFIG_PATH})"
    )
    parser.add_argument(
        "--system_config",
        type=str,
        default=DEFAULT_SYSTEM_CONFIG_PATH,
        help=f"Path to the system configuration file (default: {DEFAULT_SYSTEM_CONFIG_PATH})"
    )
    args = parser.parse_args()

    run_evaluation(
        questions_file=args.questions_csv,
        api_config_file=args.api_config,
        system_config_file=args.system_config
    )


if __name__ == "__main__":
    main()