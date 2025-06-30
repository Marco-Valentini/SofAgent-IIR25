from typing import Any, Union, List
from loguru import logger

from sofagent.agents.base import ToolAgent
from sofagent.tools import InfoFurnbase # Tool for accessing furniture data
from sofagent.utils import read_json, get_rm, parse_action, format_step


class FurnSearcher(ToolAgent):
    """
    Agent specialized in searching for non-sofa furniture items based on
    category and specified criteria (features, price, or a combination).
    It translates natural language search requests into tool commands.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """
        Initializes the FurnSearcher agent.
        Args:
            config_path (str): Path to the agent's JSON configuration file.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config) # Load and initialize tools specified in config
        self.max_turns = get_rm(config, 'max_turns', 2)  # Max internal LLM calls; typically 1 tool then Finish
        self.furn_searcher = self.get_LLM(config=config) # LLM instance for this agent
        self.json_mode = self.furn_searcher.json_mode
        self.valid_categories = [  # List of valid furniture categories for validation
            "ArrediVari", "ArteParete", "Cassettiere", "Comodini",
            "CredenzeMobiliContenitori", "Lampade", "Librerie", "MaterassiGuanciali",
            "OggettiDecorativi", "Profumatori", "SediePoltroncine", "Specchi",
            "Tappeti", "TavoliniCaffe", "TavoliPranzo", "Tessili"
        ]
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        """Specifies the tools required by this agent."""
        return {'info_furn': InfoFurnbase}

    @property
    def info_retriever(self) -> InfoFurnbase:
        """Provides access to the InfoFurnbase tool instance."""
        return self.tools['info_furn']

    @property
    def furn_searcher_prompt(self) -> str:
        """Returns the appropriate prompt template based on JSON mode."""
        return self.prompts['furn_searcher_prompt_json'] if self.json_mode else self.prompts['furn_searcher_prompt']

    @property
    def furn_searcher_examples(self) -> str:
        """Returns example interactions for the LLM, based on JSON mode."""
        return self.prompts['furn_searcher_examples_json'] if self.json_mode else self.prompts['furn_searcher_examples']

    @property
    def furn_searcher_fewshot(self) -> str:
        """Returns few-shot examples for the LLM, based on JSON mode."""
        return self.prompts['furn_searcher_fewshot_json'] if self.json_mode else self.prompts['furn_searcher_fewshot']

    @property
    def hint(self) -> str:
        """Returns a hint message, typically used when the LLM needs to Finish."""
        return self.prompts.get('furn_searcher_hint', '')

    def _build_furn_searcher_prompt(self, manager_request_str: str) -> str:
        """
        Builds the full prompt string to be sent to the FurnSearcher's LLM.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The fully formatted prompt.
        """
        return self.furn_searcher_prompt.format(
            manager_request=manager_request_str,
            history=self.history, # Formatted history of previous commands/observations
            hint=self.hint if len(self._history) + 1 >= self.max_turns else ''
        )

    def _prompt_furn_searcher(self, manager_request_str: str) -> str:
        """
        Sends the built prompt to the LLM and gets its proposed command.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The command string proposed by the LLM.
        """
        prompt = self._build_furn_searcher_prompt(
            manager_request_str=manager_request_str
        )
        command = self.furn_searcher(prompt)
        return command

    def command(self, command_str: str) -> None:
        """
        Parses the LLM's command string, validates arguments, executes the corresponding
        tool action (if any), and records the observation.
        Args:
            command_str (str): The command string from the LLM.
        """
        logger.debug(f'FurnSearcher LLM proposed command: {command_str}')
        log_head = ''
        action_type, argument = parse_action(command_str, json_mode=self.json_mode)
        observation = ""
        category_payload = ""
        condition_payload = "" # For search conditions

        valid_payload_for_tool = False # Flag to check if arguments are valid for a tool call

        # Parse and validate arguments based on the action type
        if action_type.lower() == 'listfurnfeatures':
            if self.json_mode:
                if isinstance(argument, str) and argument.strip():
                    category_payload = argument.strip(); valid_payload_for_tool = True
                else:
                    observation = f"Invalid argument for ListFurnFeatures (JSON). Expected category string. Got: {argument}"
            else: # Non-JSON mode
                if isinstance(argument, str) and argument.strip():
                    category_payload = argument.strip(); valid_payload_for_tool = True
                else:
                    observation = f"Invalid argument for ListFurnFeatures. Expected category string. Got: {argument}"
        elif action_type.lower() in ['furnfeaturequery', 'furnpricequery', 'furncomboquery']:
            if self.json_mode:
                if isinstance(argument, list) and len(argument) == 2 and isinstance(argument[0], str) and isinstance(
                        argument[1], str):
                    category_payload, condition_payload = argument[0].strip(), argument[1].strip();
                    valid_payload_for_tool = True
                else:
                    observation = f"Invalid argument for {action_type} (JSON). Expected [category, condition]. Got: {argument}"
            else: # Non-JSON mode
                if isinstance(argument, str) and ',' in argument: # Expects "category,condition"
                    parts = argument.split(',', 1)
                    if len(parts) == 2:
                        category_payload, condition_payload = parts[0].strip(), parts[
                            1].strip(); valid_payload_for_tool = True
                    else:
                        observation = f"Invalid argument format for {action_type}. Expected 'category,condition'. Got: {argument}"
                else:
                    observation = f"Invalid argument for {action_type}. Expected 'category,condition'. Got: {argument}"
        elif action_type.lower() == 'finish':
            valid_payload_for_tool = True # No tool call, but action is valid
            pass # Handled later
        else:
            observation = f'Unknown command type from FurnSearcher LLM: {action_type}.'

        # Validate category if a tool call is intended
        if valid_payload_for_tool and action_type.lower() != 'finish' and category_payload:
            if category_payload not in self.valid_categories:
                from difflib import get_close_matches # Attempt to correct minor category misspellings
                close_matches = get_close_matches(category_payload, self.valid_categories, n=1, cutoff=0.7)
                if close_matches:
                    original_cat_debug = category_payload
                    category_payload = close_matches[0]
                    logger.warning(
                        f"FurnSearcher category '{original_cat_debug}' mapped to '{category_payload}' by agent.")
                    # Potentially add this mapping info to observation if it helps Manager
                    observation = f"Original category '{original_cat_debug}' was interpreted as '{category_payload}'. Proceeding with '{category_payload}' for the tool call."
                else:
                    observation = f"Error: Category '{category_payload}' is not valid. Please use one of: {', '.join(self.valid_categories)}."
                    valid_payload_for_tool = False # Invalidate tool call if category is bad

        if not observation and valid_payload_for_tool: # If no parsing or category error and action is valid
            # Determine how to display the condition in logs (avoid showing empty quotes for random searches)
            log_cond_display = condition_payload if condition_payload.strip() and condition_payload.strip() not in [
                "''", '""'] else "<empty_query_for_random>"

            if action_type.lower() == 'listfurnfeatures':
                try:
                    observation = self.info_retriever.list_furn_features_for_category(category=category_payload)
                    log_head = f':violet[Tool Call: ListFurnFeatures for category] :blue[{category_payload}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling ListFurnFeatures for '{category_payload}': {e}"; logger.error(
                        f"ListFurnFeatures error: {e}")
            elif action_type.lower() == 'furnfeaturequery':
                try:
                    observation = self.info_retriever.furn_feature_query(category=category_payload,
                                                                         condition_str=condition_payload)
                    log_head = f':violet[Tool Call: FurnFeatureQuery for Cat]: :blue[{category_payload}], :violet[Cond]: :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling FurnFeatureQuery for '{category_payload}': {e}"; logger.error(
                        f"FurnFeatureQuery error: {e}")
            elif action_type.lower() == 'furnpricequery':
                try:
                    observation = self.info_retriever.furn_price_query(category=category_payload,
                                                                       price_condition_str=condition_payload)
                    log_head = f':violet[Tool Call: FurnPriceQuery for Cat]: :blue[{category_payload}], :violet[Cond]: :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling FurnPriceQuery for '{category_payload}': {e}"; logger.error(
                        f"FurnPriceQuery error: {e}")
            elif action_type.lower() == 'furncomboquery':
                try:
                    observation = self.info_retriever.furn_combo_query(category=category_payload,
                                                                       complex_condition_str=condition_payload)
                    log_head = f':violet[Tool Call: FurnComboQuery for Cat]: :blue[{category_payload}], :violet[Cond]: :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling FurnComboQuery for '{category_payload}': {e}"; logger.error(
                        f"FurnComboQuery error: {e}")
            elif action_type.lower() == 'finish':
                observation = self.finish(results=argument) # Argument is the result string for the Manager
                log_head = ':violet[Finish with results for Manager]:\n- '
            # Removed else for unknown, already handled

        logger.debug(f'FurnSearcher Observation: {observation}')
        self.observation(str(observation), log_head) # Logs to system (and Streamlit if web_demo)
        turn = {'command': command_str, 'observation': str(observation)};
        self._history.append(turn)

    def forward(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """
        Main execution loop for the FurnSearcher. Calls its LLM to decide on an
        action (tool call or Finish) and processes it. Typically one tool call then Finish.
        Args:
            manager_request (str): The initial request from the Manager.
        Returns:
            str: The final result to be passed back to the Manager.
        """
        self._history = [] # Clear history for this specific sub-task

        while not self.is_finished():
            command_str = self._prompt_furn_searcher(manager_request_str=manager_request)
            self.command(command_str)
            # Prompt design expects Finish after one tool call or if listing features.

        if not self.finished: # If max_turns reached without Finish
            logger.warning("FurnSearcher did not finish within max_turns.")
            if self._history: return self._history[-1]['observation'] # Return last observation
            return "FurnSearcher sub-task did not complete with a Finish command or provide any observation."

        return self.results if self.results is not None else "No results from FurnSearcher."

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Entry point for the system to call this agent.
        Args:
            argument (Any): The natural language request from the Manager.
            json_mode (bool): Indicates if the LLM should operate in JSON mode.
        Returns:
            str: The final result from the agent's processing.
        """
        self.reset();
        self.json_mode = json_mode
        if not isinstance(argument, str) or not argument.strip():
            return "Invalid argument for FurnSearcher. Expected a natural language request string from Manager."
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
        agent_prompts_file = os.path.join(project_root, 'config/prompts/agent_prompt/furn_searcher.json')
        agent_config_file = os.path.join(project_root, 'config/agents/furn_searcher.json')

        init_api(read_json(api_config_file))
        prompts = read_prompts(agent_prompts_file)

        # Dummy system class for testing purposes
        class DummySystem:
            def __init__(self):
                self.web_demo = False; self.task_type = "search_furniture"; self.data_sample = None

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
        furn_searcher = FurnSearcher(config_path=agent_config_file, prompts=prompts, system=dummy_sys)

        # Example test requests
        test_requests_from_manager = [
            "For the Lampade category, what features can I search by and what are their values?",
            "What are the searchable attributes for Tappeti?",
            "Tell me about search options for InvalidCategory.",
            "Find floor lamps (tipo_lampada=Piantana) made of metal (materiale_struttura=Metallo) in the Lampade category.",
            "Search for rectangular rugs (forma=Rettangolare) in the Tappeti category.",
            "I'm looking for chairs (category SediePoltroncine) with arms (tipo_sedia=Con Braccioli).",
            "In Lampade, find items where NonExistentFeature=Foo.",
            "In TavoliniCaffe, find items with an invalid operator: forma ??? Rotondo",
            "In TavoliniCaffe, find items with a type mismatch: prezzo > abc",
            "Show me coffee tables from TavoliniCaffe that cost less than 600 euros.",
            "Find chairs in SediePoltroncine with a price between 800 and 1200.",
            "I need armchairs from SediePoltroncine in leather (materiale_rivestimento_seduta CONTAINS Pelle) that cost less than 1000 euros (prezzo < 1000).",
            "Find mirrors (category Specchi) with a metal frame (materiale_cornice=Metallo) and price above 100 (prezzo > 100).",
            "Show me 5 random items from the Lampade category.",
            "Can you suggest some Tappeti?",
            "Tell me about some OggettiDecorativi."
        ]
        # Loop through test requests and invoke the agent
        for i, manager_req in enumerate(test_requests_from_manager):
            logger.info(f"--- FurnSearcher Test Case {i + 1} ---")
            logger.info(f"Manager Request to FurnSearcher: \"{manager_req}\"")
            response_to_manager = furn_searcher.invoke(argument=manager_req, json_mode=furn_searcher.json_mode)
            logger.success(f"FurnSearcher's Response (to Manager):\n{response_to_manager}\n")
            furn_searcher.reset() # Reset agent state for next test
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}. CWD: {os.getcwd()}. Root: {project_root}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)