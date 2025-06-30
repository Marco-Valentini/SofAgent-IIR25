from typing import Any, Union, List
from loguru import logger

from sofagent.agents.base import ToolAgent
from sofagent.tools import InfoSofabase  # Tool for accessing sofa-specific data
from sofagent.utils import read_json, get_rm, parse_action, format_step


class SofaSearcher(ToolAgent):
    """
    Agent specialized in searching for sofas based on various criteria
    such as features, price, number of seats, or a combination thereof.
    It translates natural language search requests into tool commands.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """
        Initializes the SofaSearcher agent.
        Args:
            config_path (str): Path to the agent's JSON configuration file.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config) # Load and initialize tools (InfoSofabase)
        self.max_turns = get_rm(config, 'max_turns',
                                2)  # Max internal LLM calls; typically 1 tool call then Finish
        self.llm = self.get_LLM(config=config) # LLM instance for this agent
        self.json_mode = self.llm.json_mode # Whether the LLM expects/outputs JSON
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        """Specifies the tools required by this agent."""
        return {'info_sofa': InfoSofabase}

    @property
    def info_retriever(self) -> InfoSofabase:
        """Provides access to the InfoSofabase tool instance."""
        return self.tools['info_sofa']

    @property
    def sofa_searcher_prompt_template(self) -> str:
        """Returns the main prompt template for the SofaSearcher, choosing JSON version if applicable."""
        return self.prompts['sofa_searcher_prompt_json'] if self.json_mode else self.prompts['sofa_searcher_prompt']

    @property
    def hint_prompt(self) -> str:
        """Returns a hint message, typically used when the LLM needs to Finish."""
        return self.prompts.get('sofa_searcher_hint', '')

    def _build_prompt(self, manager_request_str: str) -> str:
        """
        Builds the full prompt string to be sent to the SofaSearcher's LLM.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The fully formatted prompt.
        """
        # The prompt content now includes the feature list directly, so no separate 'features_list_for_prompt' is needed here.
        return self.sofa_searcher_prompt_template.format(
            manager_request=manager_request_str,
            history=self.history, # Formatted history of previous commands/observations
            hint=self.hint_prompt if len(self._history) + 1 >= self.max_turns else ''
        )

    def _prompt_sofa_searcher(self, manager_request_str: str) -> str:
        """
        Sends the built prompt to the LLM and gets its proposed command.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The command string proposed by the LLM.
        """
        prompt = self._build_prompt(manager_request_str=manager_request_str)
        command_from_llm = self.llm(prompt)
        return command_from_llm

    def command(self, command_str: str) -> None:
        """
        Parses the LLM's command string, executes the corresponding tool action,
        and records the observation. The LLM is expected to decide if it's providing
        searchable features (via Finish) or making a tool call.
        Args:
            command_str (str): The command string from the LLM.
        """
        logger.debug(f'SofaSearcher LLM proposed command: {command_str}')
        log_head = '' # For formatting Streamlit logs
        action_type, argument = parse_action(command_str, json_mode=self.json_mode)
        observation = ""

        condition_payload = ""  # Default for tool calls requiring a condition string
        if isinstance(argument, str):
            condition_payload = argument
        elif isinstance(argument, list) and len(argument) == 1 and isinstance(argument[0], str):  # Handles JSON: "content": ["condition_string"]
            condition_payload = argument[0]
        elif argument is None and action_type.lower() == 'listsofafeatures':
            # This action was removed as its functionality is now integrated into the prompt or handled by Finish
            # If it were still a distinct action, this check would be relevant.
            pass
        elif action_type.lower() != 'finish':  # If not a Finish action and payload isn't a simple string
            observation = f"Invalid argument format for {action_type}. Expected a condition string. Received: {argument}"

        # For logging, display "<empty_query_for_random>" if condition is empty, otherwise the condition
        log_cond_display = condition_payload if condition_payload.strip() and condition_payload.strip() not in ["''", '""'] else "<empty_query_for_random>"

        if not observation:  # Proceed if no parsing error
            if action_type.lower() == 'sofafeaturequery':
                try:
                    observation = self.info_retriever.sofa_feature_query(condition_str=condition_payload)
                    log_head = f':violet[Tool Call: SofaFeatureQuery with conditions] :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling SofaFeatureQuery: {e}";
                    logger.error(f"SofaFeatureQuery error: {e}")
            elif action_type.lower() == 'sofapricequery':
                try:
                    observation = self.info_retriever.sofa_price_query(price_condition_str=condition_payload)
                    log_head = f':violet[Tool Call: SofaPriceQuery with conditions] :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling SofaPriceQuery: {e}";
                    logger.error(f"SofaPriceQuery error: {e}")
            elif action_type.lower() == 'sofaseatsquery':
                try:
                    observation = self.info_retriever.sofa_seats_query(seats_condition_str=condition_payload)
                    log_head = f':violet[Tool Call: SofaSeatsQuery with conditions] :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling SofaSeatsQuery: {e}";
                    logger.error(f"SofaSeatsQuery error: {e}")
            elif action_type.lower() == 'sofacomboquery':
                try:
                    observation = self.info_retriever.sofa_combo_query(complex_condition_str=condition_payload)
                    log_head = f':violet[Tool Call: SofaComboQuery with conditions] :red[{log_cond_display}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling SofaComboQuery: {e}";
                    logger.error(f"SofaComboQuery error: {e}")
            elif action_type.lower() == 'finish':
                # If the Manager asked "what features can I search by?",
                # the LLM of this agent is prompted to formulate the feature list and use Finish.
                # In this case, `argument` will be that formatted string of features.
                observation = self.finish(results=argument)
                log_head = ':violet[Finish with results for Manager]:\n- '
            else:
                # Handling for a potential 'ListSofaFeatures' action if it were distinct (now integrated).
                # If an unknown action is received, it's logged as such.
                observation = f'Unknown command type from SofaSearcher LLM: {action_type}. The Manager asked for features, and I should have used Finish.' if 'listsofafeatures' in action_type.lower() else f'Unknown command type from SofaSearcher LLM: {action_type}.'

        logger.debug(f'SofaSearcher Observation: {observation}')
        self.observation(str(observation), log_head) # Logs to system (and Streamlit if web_demo)
        turn = {'command': command_str, 'observation': str(observation)}
        self._history.append(turn)

    def forward(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """
        Main execution loop for the SofaSearcher. It calls its LLM to decide on an
        action (tool call or Finish to list features/return results) and processes it.
        The agent is designed to Finish after one primary action (tool call or listing features).
        Args:
            manager_request (str): The initial request from the Manager.
        Returns:
            str: The final result (search results or feature list) to be passed back to the Manager.
        """
        self._history = []  # Clear history for this specific sub-task from Manager

        while not self.is_finished(): # Loop typically runs once or twice (e.g. one tool call, then Finish)
            if len(self._history) >= self.max_turns: # Safeguard for unexpected LLM behavior
                logger.warning(
                    f"{self.__class__.__name__} reached max_turns ({self.max_turns}) without a Finish command.")
                if self._history: # Return last known observation if max_turns is hit
                    return f"Max turns reached. Last observation: {self._history[-1]['observation']}"
                else:
                    return "Max turns reached, but no actions were taken or observations made."

            current_manager_request_for_prompt = manager_request

            command_str = self._prompt_sofa_searcher(manager_request_str=current_manager_request_for_prompt)
            self.command(command_str)  # This appends to self.history and will set self.finished if LLM chooses Finish

            if self.is_finished():  # Exit loop if Finish command was executed
                break

        if not self.finished: # Should ideally not be hit if prompt is followed
            logger.error(
                f"{self.__class__.__name__} exited loop but was not marked as finished and has no history. This should not happen.")
            return "An unexpected state occurred in the agent's execution."

        return self.results if self.results is not None else "No results were finalized by the agent."


    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Entry point for the system to call this agent.
        Args:
            argument (Any): The natural language request from the Manager.
            json_mode (bool): Indicates if the LLM should operate in JSON mode.
        Returns:
            str: The final result from the agent's processing.
        """
        self.reset(); # Reset agent state for a new invocation
        self.json_mode = json_mode
        if not isinstance(argument, str) or not argument.strip():
            return "Invalid argument for SofaSearcher. Expected a natural language request string from Manager."
        manager_request_str = argument.strip()
        return self.forward(manager_request=manager_request_str)

    def __call__(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """Allows the agent to be called like a function."""
        self.reset(); # Reset agent state
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
        agent_prompts_file = os.path.join(project_root, 'config/prompts/agent_prompt/sofa_searcher.json')
        agent_config_file = os.path.join(project_root, 'config/agents/sofa_searcher.json')

        # Manually update tool config path to ensure the new file is loaded for the test
        tool_config_path = os.path.join(project_root, 'config/tools/info_database/sofa.json')
        tool_config = read_json(tool_config_path)
        # Ensure the path to predefined configurations is present for testing seat queries
        if 'predefined_configs_info' not in tool_config:
            tool_config['predefined_configs_info'] = 'data/passport_data/ConfigurazioniPredefiniteDivani_ConPrezziESedute.csv'

        init_api(read_json(api_config_file)) # Initialize API
        prompts = read_prompts(agent_prompts_file) # Load prompt templates

        # Dummy system class for testing purposes
        class DummySystem:
            def __init__(self):
                self.web_demo = False;
                self.task_type = "search_sofas"; # Task type for prompt formatting
                self.data_sample = None

            def log(self, m, agent=None, logging=True): # Simplified logging for console
                role_name = agent.__class__.__name__ if agent else 'AgentInternal'
                if "Tool Call" in m or "Finish" in m or "Manager Request" in m:
                    logger.info(f"({role_name}): {m}")
                else:
                    logger.debug(f"({role_name}): {m}")

        dummy_sys = DummySystem()
        # Partial prompt formatting if task_type is an input variable
        for pname, ptemplate in prompts.items():
            if isinstance(ptemplate, PromptTemplate) and 'task_type' in ptemplate.input_variables:
                prompts[pname] = ptemplate.partial(task_type=dummy_sys.task_type)

        # Instantiate the SofaSearcher agent
        sofa_searcher = SofaSearcher(config_path=agent_config_file, prompts=prompts, system=dummy_sys)

        # Example test requests from the Manager
        test_requests_from_manager = [
            # # Test Case 1: Simple Feature Query (unchanged)
            "Find sofas that are modular and have adjustable headrests.",
            # # Test Case 2: Simple Price Query (unchanged)
            "Show me sofas that cost less than 4000 euros.",
            # Test Case 3: Simple Seats Query (NEW)
            "Find sofas with exactly 3 seats.",
            # Test Case 4: Advanced Seats Query (NEW)
            "I need a large sofa, show me ones with 4 or more seats.",
            # Test Case 5: Combo Query with Seats and Features (NEW)
            "I need a modular sofa that has at least 3 seats.",
            # Test Case 6: Full Combo Query with Price, Seats, and Features (NEW)
            "I'm looking for a leather sofa with 3 seats that costs less than 12000 euros.",
        ]

        # Loop through test requests and invoke the agent
        for i, manager_req in enumerate(test_requests_from_manager):
            logger.info(f"--- SofaSearcher Test Case {i + 1} ---")
            logger.info(f"Manager Request to SofaSearcher: \"{manager_req}\"")
            response_to_manager = sofa_searcher.invoke(argument=manager_req, json_mode=sofa_searcher.json_mode)
            logger.success(f"SofaSearcher's Final Response (to Manager):\n{response_to_manager}\n")
            # Agent's reset is called within invoke()

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}. CWD: {os.getcwd()}. Root: {project_root}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)