from typing import Any, Union, List, Optional
from loguru import logger

from sofagent.agents.base import ToolAgent
from sofagent.tools import InfoFurnbase
from sofagent.utils import read_json, get_rm, parse_action, format_step

class FurnAnalyzer(ToolAgent):
    """
    Agent specialized in analyzing and retrieving detailed information about
    non-sofa furniture items (furnishings, accessories, etc.).
    It uses the InfoFurnbase tool to query structured data.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """
        Initializes the FurnAnalyzer agent.
        Args:
            config_path (str): Path to the agent's JSON configuration file.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config)
        self.max_turns = get_rm(config, 'max_turns', 5) # Maximum number of internal LLM calls before forcing a finish
        self.furn_analyzer = self.get_LLM(config=config) # LLM instance for this agent
        self.json_mode = self.furn_analyzer.json_mode
        self.valid_categories = [
            "ArrediVari", "ArteParete", "Cassettiere", "Comodini",
            "CredenzeMobiliContenitori", "Lampade", "Librerie", "MaterassiGuanciali",
            "OggettiDecorativi", "Profumatori", "SediePoltroncine", "Specchi",
            "Tappeti", "TavoliniCaffe", "TavoliPranzo", "Tessili"
        ]
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        """Specifies the tools required by this agent."""
        return {
            'info_furn': InfoFurnbase
        }

    @property
    def info_retriever(self) -> InfoFurnbase:
        """Provides access to the InfoFurnbase tool instance."""
        return self.tools['info_furn']

    @property
    def furn_analyzer_prompt(self) -> str:
        """Returns the appropriate prompt template based on JSON mode."""
        if self.json_mode:
            return self.prompts['furn_analyzer_prompt_json']
        else:
            return self.prompts['furn_analyzer_prompt']

    @property
    def furn_analyzer_examples(self) -> str:
        """Returns example interactions for the LLM, based on JSON mode."""
        if self.json_mode:
            return self.prompts['furn_analyzer_examples_json']
        else:
            return self.prompts['furn_analyzer_examples']

    @property
    def furn_analyzer_fewshot(self) -> str:
        """Returns few-shot examples for the LLM, based on JSON mode."""
        if self.json_mode:
            return self.prompts['furn_analyzer_fewshot_json']
        else:
            return self.prompts['furn_analyzer_fewshot']

    @property
    def hint(self) -> str:
        """Returns a hint message, typically used when nearing max_turns."""
        if 'furn_analyzer_hint' not in self.prompts:
            return ''
        return self.prompts['furn_analyzer_hint']

    def _build_furn_analyzer_prompt(self, manager_request_str: str) -> str:
        """
        Builds the full prompt string to be sent to the FurnAnalyzer's LLM.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The fully formatted prompt.
        """
        return self.furn_analyzer_prompt.format(
            manager_request=manager_request_str,
            history=self.history, # Formatted history of previous commands/observations in this sub-task
            hint=self.hint if len(self._history) + 1 >= self.max_turns else ''
        )

    def _prompt_furn_analyzer(self, manager_request_str: str) -> str:
        """
        Sends the built prompt to the LLM and gets its proposed command.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The command string proposed by the LLM.
        """
        furn_analyzer_prompt = self._build_furn_analyzer_prompt(
            manager_request_str=manager_request_str
        )
        command = self.furn_analyzer(furn_analyzer_prompt)
        return command

    def command(self, command_str: str) -> None:
        """
        Parses the LLM's command string, executes the corresponding tool action,
        and records the observation.
        Args:
            command_str (str): The command string from the LLM.
        """
        logger.debug(f'FurnAnalyzer LLM proposed command: {command_str}')
        log_head = '' # For formatting Streamlit logs
        action_type, argument = parse_action(command_str, json_mode=self.json_mode)
        observation = ""

        category_payload = ""
        item_query_payload = ""

        # Validate argument structure for tool calls
        if not isinstance(argument, list) and action_type.lower() != 'finish':
            observation = f"Invalid argument structure for {action_type}. Expected a list."
        elif action_type.lower() != 'finish' and len(argument) != 2 and action_type.lower() == 'getfurnid':
            observation = f"Invalid argument count for {action_type}. Expected [category, name]. Got {len(argument)} elements."
        elif action_type.lower() != 'finish' and len(argument) != 2 and action_type.lower() == 'furninfoandprice':
            observation = f"Invalid argument count for {action_type}. Expected [category, id_or_name_string]. Got {len(argument)} elements."
        elif action_type.lower() != 'finish': # If arguments seem structurally okay for tool calls
            category_payload = str(argument[0]).strip()
            item_query_payload = str(argument[1]).strip()

            if category_payload not in self.valid_categories:
                from difflib import get_close_matches # Attempt to correct minor category misspellings
                close_matches = get_close_matches(category_payload, self.valid_categories, n=1, cutoff=0.7)
                if close_matches:
                    original_cat = category_payload
                    category_payload = close_matches[0]
                    logger.warning(
                        f"FurnAnalyzer category '{original_cat}' mapped to '{category_payload}' for tool call.")
                else:
                    observation = f"Error: Category '{category_payload}' is not a valid furniture category. Please use one of: {', '.join(self.valid_categories)}."

        if not observation:  # Proceed if no parsing or category validation error
            if action_type.lower() == 'getfurnid':
                if category_payload and item_query_payload:
                    try:
                        observation = self.info_retriever.get_furn_id_by_name(category=category_payload,
                                                                              furn_name=item_query_payload)
                        log_head = f':violet[Tool Call: GetFurnID for] :blue[{category_payload}] :red[{item_query_payload}]:violet[...]\n- '
                    except Exception as e:
                        observation = f"Error calling GetFurnID for '{item_query_payload}' in '{category_payload}': {e}";
                        logger.error(f"GetFurnID error: {e}")
                else:
                    observation = "Missing category or item name for GetFurnID."

            elif action_type.lower() == 'furninfoandprice':
                # LLM provides a single category and a string of items (comma-separated or single) for that category.
                items_for_category = [item.strip() for item in item_query_payload.split(',')]
                # Tool expects list of [category, item] pairs, so we construct it for the current category.
                category_item_pairs = [[category_payload, item] for item in items_for_category if item]

                if category_payload and category_item_pairs:
                    try:
                        observation = self.info_retriever.furn_info_and_price(category_items=category_item_pairs)
                        log_head = f':violet[Tool Call: FurnInfoAndPrice for category] :blue[{category_payload}] :violet[items] :red[{items_for_category}]:violet[...]\n- '
                    except Exception as e:
                        observation = f"Error calling FurnInfoAndPrice for items in '{category_payload}': {e}";
                        logger.error(f"FurnInfoAndPrice error: {e}")
                else:
                    observation = "Missing category or item details for FurnInfoAndPrice."

            elif action_type.lower() == 'finish':
                observation = self.finish(results=argument) # Argument is the result string for the Manager
                log_head = ':violet[Finish with results for Manager]:\n- '
            else:
                observation = f'Unknown command type from FurnAnalyzer LLM: {action_type}.'

        logger.debug(f'FurnAnalyzer Observation: {observation}')
        self.observation(str(observation), log_head) # Logs to system (and Streamlit if web_demo)
        turn = {'command': command_str, 'observation': str(observation)}
        self._history.append(turn)

    def forward(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """
        Main execution loop for the FurnAnalyzer. It iteratively calls its LLM
        to decide on actions (tool calls) until it calls Finish or hits max_turns.
        Args:
            manager_request (str): The initial request from the Manager.
        Returns:
            str: The final result to be passed back to the Manager.
        """
        self._history = [] # Clear history for this specific sub-task

        while not self.is_finished():
            command_str = self._prompt_furn_analyzer(manager_request_str=manager_request)
            self.command(command_str)
            # Prompt design encourages LLM to use Finish after a tool call if request is satisfied.

        if not self.finished: # If max_turns reached without Finish
            logger.warning("FurnAnalyzer did not finish within max_turns.")
            if self._history: # Return last observation if any, might contain partial data or an error message
                return self._history[-1]['observation']
            return "FurnAnalyzer sub-task did not complete with a Finish command or provide any observation."

        return self.results if self.results is not None else "No results from FurnAnalyzer."

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Entry point for the system to call this agent.
        Args:
            argument (Any): The natural language request from the Manager.
            json_mode (bool): Indicates if the LLM should operate in JSON mode.
        Returns:
            str: The final result from the agent's processing.
        """
        self.reset()
        self.json_mode = json_mode
        if not isinstance(argument, str) or not argument.strip():
            return "Invalid argument for FurnAnalyzer. Expected a natural language request string from Manager."
        manager_request_str = argument.strip()
        return self.forward(manager_request=manager_request_str)

    def __call__(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """Allows the agent to be called like a function."""
        self.reset();
        return self.forward(manager_request=manager_request, *args, **kwargs)


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from sofagent.utils import init_api, read_json, read_prompts
    import sys
    import os

    # --- Setup for local script execution ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    try:
        # Define paths to configuration files
        api_config_file = os.path.join(project_root, 'config/api-config.json')
        agent_prompts_file = os.path.join(project_root, 'config/prompts/agent_prompt/furn_analyzer.json')
        agent_config_file = os.path.join(project_root, 'config/agents/furn_analyzer.json')

        # Initialize API and load prompts
        init_api(read_json(api_config_file))
        prompts = read_prompts(agent_prompts_file)

        # Dummy system class for testing purposes
        class DummySystem:
            def __init__(self):
                self.web_demo = False; self.task_type = "analyze_furniture"; self.data_sample = None

            def log(self, m, agent=None, logging=True):
                role_name = agent.__class__.__name__ if agent else 'AgentInternal'
                if "Tool Call" in m or "Finish" in m or "Manager Request" in m:
                    logger.info(f"({role_name}): {m}")
                else:
                    logger.debug(f"({role_name}): {m}")

        dummy_sys = DummySystem()
        # Partial prompts if they expect task_type
        for pname, ptemplate in prompts.items():
            if isinstance(ptemplate, PromptTemplate) and 'task_type' in ptemplate.input_variables:
                prompts[pname] = ptemplate.partial(task_type=dummy_sys.task_type)

        # Instantiate the agent
        furn_analyzer = FurnAnalyzer(config_path=agent_config_file, prompts=prompts, system=dummy_sys)

        # Example test requests
        test_requests_from_manager = [
            "What is the ID for the 'Circle' lamp in the Lampade category?",
            "Find the product code for the 'Ido' coffee table, category TavoliniCaffe.",
            "I need the ID of 'Kiris' chair from SediePoltroncine.",
            "What's the ID for 'NonExistentLamp' in Lampade category?",
            "Get ID for 'Ido' in an InvalidCategory.",
            "Tell me about lamp L50601G from Lampade category, including its price.",
            "Provide features and price for the coffee table named 'CABARET' in TavoliniCaffe category.",
            "I want details and price for items 'R87001X' and 'Papel' from the Tappeti category.",
            "Describe the 'AMBRA' item from SediePoltroncine and its cost.",
            "Get info and price for furniture L51411G in Lampade category.",
            "Info and price for 'NonExistentID123' in Specchi category."
        ]
        # Loop through test requests and invoke the agent
        for i, manager_req in enumerate(test_requests_from_manager):
            logger.info(f"--- FurnAnalyzer Test Case {i + 1} ---")
            logger.info(f"Manager Request to FurnAnalyzer: \"{manager_req}\"")
            response_to_manager = furn_analyzer.invoke(argument=manager_req, json_mode=furn_analyzer.json_mode)
            logger.success(f"FurnAnalyzer's Response (to Manager):\n{response_to_manager}\n")
            furn_analyzer.reset() # Reset agent state for next test
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}. CWD: {os.getcwd()}. Root: {project_root}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)