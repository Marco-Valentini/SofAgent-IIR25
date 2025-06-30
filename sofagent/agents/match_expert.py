from typing import Any, Union, List, Optional, Dict
from loguru import logger

from sofagent.agents.base import ToolAgent
from sofagent.tools import InfoMatchbase # Tool for accessing matching-related data
from sofagent.utils import read_json, get_rm, parse_action, format_step


class MatchExpert(ToolAgent):
    """
    Agent specialized in finding and suggesting harmonious pairings between furniture items.
    It can perform generic stylistic matches, color-based matches, and layout-based matches.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """
        Initializes the MatchExpert agent.
        Args:
            config_path (str): Path to the agent's JSON configuration file.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config) # Load and initialize tools (InfoMatchbase)
        self.max_turns = get_rm(config, 'max_turns', 2)  # Usually 1 tool call then Finish
        self.llm = self.get_LLM(config=config) # LLM instance for this agent
        self.json_mode = self.llm.json_mode
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        """Specifies the tools required by this agent."""
        return {'info_match': InfoMatchbase}

    @property
    def info_retriever(self) -> InfoMatchbase:
        """Provides access to the InfoMatchbase tool instance."""
        return self.tools['info_match']

    @property
    def match_expert_prompt_template(self) -> str:
        """Returns the appropriate prompt template based on JSON mode."""
        return self.prompts['match_expert_prompt_json'] if self.json_mode else self.prompts['match_expert_prompt']

    @property
    def hint_prompt(self) -> str:
        """Returns a hint message, typically used when the LLM needs to Finish."""
        return self.prompts.get('match_expert_hint', '')

    def _build_prompt(self, manager_request_str: str) -> str:
        """
        Builds the full prompt string to be sent to the MatchExpert's LLM.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The fully formatted prompt.
        """
        return self.match_expert_prompt_template.format(
            manager_request=manager_request_str,
            history=self.history, # Formatted history of previous commands/observations in this sub-task
            hint=self.hint_prompt if len(self._history) + 1 >= self.max_turns else ''
        )

    def _prompt_llm(self, manager_request_str: str) -> str:
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

    def _format_matches_for_llm(self, matches: List[Dict[str, Any]], match_type_invoked: str) -> str:
        """
        Formats the list of match dictionaries into a string suitable for the LLM's observation.
        Args:
            matches (List[Dict[str, Any]]): List of match dictionaries from the tool.
            match_type_invoked (str): The type of match action that was called (e.g., "GenericMatch").
        Returns:
            str: A formatted string summarizing the matches or error/info messages.
        """
        if not matches:
            return f"No matches found via {match_type_invoked}."
        # Handle specific error or info messages from the tool
        if isinstance(matches, list) and matches and "error" in matches[0]:
            return f"Tool Error ({match_type_invoked}): {matches[0]['error']}"
        if isinstance(matches, list) and matches and "info" in matches[0]:
            return f"Tool Info ({match_type_invoked}): {matches[0]['info']}"

        formatted_strings = []
        for match in matches:
            desc = f"Type: {match.get('type', 'N/A')}, ID: {match.get('id', 'N/A')}, Name: {match.get('name', 'N/A')}"
            if match.get('category'): desc += f", Category: {match.get('category')}"
            if match.get('motivation'): desc += f" (Motivation: {match.get('motivation')})"
            # Include layout-specific details if present
            if 'original_layout_role' in match:
                desc += f" (Layout Role: {match['original_layout_role']}, Config in Layout: {match.get('original_layout_config', 'N/A')})"
            formatted_strings.append(desc)

        if not formatted_strings: return f"No valid match data to format from {match_type_invoked}."
        return f"Matches found via {match_type_invoked}:\n" + "\n".join(formatted_strings)

    def command(self, command_str: str) -> None:
        """
        Parses the LLM's command string, validates arguments, executes the corresponding
        tool action (if any), and records the observation.
        Args:
            command_str (str): The command string from the LLM.
        """
        logger.debug(f'MatchExpert LLM proposed command: {command_str}')
        log_head = '' # For formatting Streamlit logs
        action_type, argument = parse_action(command_str, json_mode=self.json_mode)
        observation = ""

        category_arg, item_id_arg, optional_arg = "", "", None # Initialize arguments for tool calls
        valid_args_for_tool_call = False # Flag to check if arguments are valid for a tool call

        # Validate arguments based on the action type
        if action_type.lower() == 'listmatchingcapabilities':
            valid_args_for_tool_call = True
        elif action_type.lower() != 'finish': # For tool-using actions
            if not isinstance(argument, list):
                observation = f"Invalid argument structure for {action_type}. Expected a list."
            elif len(argument) < 2: # All match types require at least type_string and item_id_string
                observation = f"Insufficient arguments for {action_type}. Expected at least [category, item_id]."
            else:
                category_arg = str(argument[0]).strip()
                item_id_arg = str(argument[1]).strip()
                if len(argument) > 2 and argument[2] is not None and str(argument[2]).strip():
                    optional_arg = str(argument[2]).strip() # For color_theme or space_constraint

                # Validate category_arg ('sofa' or 'furn')
                if category_arg.lower() not in ["sofa", "furn"]:
                    observation = f"Invalid category '{category_arg}' for {action_type}. Must be 'sofa' or 'furn'."
                elif not item_id_arg: # item_id_string cannot be empty
                    observation = f"Item ID cannot be empty for {action_type}."
                else:
                    valid_args_for_tool_call = True
        else:  # Finish action
            valid_args_for_tool_call = True

        # Proceed with tool call or Finish if no parsing errors
        if not observation and valid_args_for_tool_call:
            if action_type.lower() == 'listmatchingcapabilities':
                try:
                    observation = self.info_retriever.list_matching_capabilities()
                    log_head = ':violet[Tool Call: ListMatchingCapabilities]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling ListMatchingCapabilities: {e}";
                    logger.error(f"ListMatchingCapabilities error: {e}")
            elif action_type.lower() == 'genericmatch':
                try:
                    matches = self.info_retriever.get_matches_from_abbinamenti(category_arg, item_id_arg)
                    observation = self._format_matches_for_llm(matches, "GenericMatch")
                    log_head = f':violet[Tool Call: GenericMatch for {category_arg}] :red[{item_id_arg}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error: {e}"; logger.error(f"GenericMatch error: {e}")
            elif action_type.lower() == 'colormatch':
                try:
                    matches = self.info_retriever.get_matches_from_moodboard(category_arg, item_id_arg, optional_arg)
                    observation = self._format_matches_for_llm(matches, "ColorMatch")
                    log_head = f':violet[Tool Call: ColorMatch for {category_arg}] :red[{item_id_arg}] (Theme: {optional_arg or "any"}):violet[...]\n- '
                except Exception as e:
                    observation = f"Error: {e}"; logger.error(f"ColorMatch error: {e}")
            elif action_type.lower() == 'layoutmatch':
                try:
                    matches = self.info_retriever.get_matches_from_layout(category_arg, item_id_arg, optional_arg)
                    observation = self._format_matches_for_llm(matches, "LayoutMatch")
                    log_head = f':violet[Tool Call: LayoutMatch for {category_arg}] :red[{item_id_arg}] (Space: {optional_arg or "any"}):violet[...]\n- '
                except Exception as e:
                    observation = f"Error: {e}"; logger.error(f"LayoutMatch error: {e}")
            elif action_type.lower() == 'finish':
                observation = self.finish(results=argument) # Argument is the result string for the Manager
                log_head = ':violet[Finish with results for Manager]:\n- '
            else:
                if not observation: observation = f'Unknown command type from MatchExpert LLM: {action_type}.'

        logger.debug(f'MatchExpert Observation: {observation}')
        self.observation(str(observation), log_head) # Logs to system (and Streamlit if web_demo)
        turn = {'command': command_str, 'observation': str(observation)}
        self._history.append(turn)

    def forward(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """
        Main execution loop for the MatchExpert. It calls its LLM to decide on an
        action (tool call or Finish) and processes it. Typically one tool call then Finish.
        Args:
            manager_request (str): The initial request from the Manager.
        Returns:
            str: The final result to be passed back to the Manager.
        """
        self._history = [] # Clear history for this specific sub-task
        while not self.is_finished():
            command_str = self._prompt_llm(manager_request_str=manager_request)
            self.command(command_str) # This appends to self.history and might set self.finished
        if not self.finished: # If max_turns reached without Finish
            logger.warning("MatchExpert did not finish. Returning last observation or error.")
            if self._history: return self._history[-1]['observation']
            return "MatchExpert sub-task did not complete with a Finish command."
        return self.results if self.results is not None else "No results from MatchExpert."

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
            return "Invalid argument for MatchExpert. Expected a natural language request string from Manager."
        return self.forward(manager_request=argument.strip())

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
        agent_prompts_file = os.path.join(project_root, 'config/prompts/agent_prompt/match_expert.json')
        agent_config_file = os.path.join(project_root, 'config/agents/match_expert.json')

        # Initialize API and load prompts
        init_api(read_json(api_config_file))
        prompts = read_prompts(agent_prompts_file)

        # Dummy system class for testing purposes
        class DummySystem:
            def __init__(self):
                self.web_demo = False
                self.task_type = "find_matches"
                self.data_sample = None

            def log(self, m, agent=None, logging=True):
                role_name = agent.__class__.__name__ if agent else 'AgentInternal'
                if "Tool Call" in m or "Finish" in m or "Manager Request" in m or "Error" in m or "Info" in m:
                    logger.info(f"({role_name}): {m}")
                else:
                    logger.debug(f"({role_name}): {m}")

        dummy_sys = DummySystem()
        # Partial prompts if they expect task_type
        for pname, ptemplate in prompts.items():
            if isinstance(ptemplate, PromptTemplate) and 'task_type' in ptemplate.input_variables:
                prompts[pname] = ptemplate.partial(task_type=dummy_sys.task_type)

        # Instantiate the agent
        match_expert = MatchExpert(config_path=agent_config_file, prompts=prompts, system=dummy_sys)

        # Example test requests
        test_requests_from_manager = [
            "What types of matching constraints do you support, and what are some example values?",
            "How can you help me find matching items? What are my options for filtering matches?",
            "What items generally match well with sofa ID 2954?",
            "Suggest generic stylistic matches for furniture ID T100MY1 (category TavoliniCaffe).",
            "Find generic matches for sofa ID 0000 (non-existent for abbinamenti).",
            "Find items that complement sofa 2954, focusing on a 'blue' color theme.",
            "What matches furniture L51402G (from Lampade category) with a 'warm neutral 1' theme from the moodboards?",
            "Color match for sofa ID 3273 and theme 'earth tone'.",
            "Suggest color matches for sofa 2957 if the theme is 'non_existent_color_theme'.",
            "What color matches are there for furniture ID Q092007XNA?",
            "What sofas and armchairs would fit well with armchair 3102:066 in a layout for a 15 sqm room?",
            "Suggest items for sofa 2957 for a 'large' space based on your layouts.",
            "Layout matches for sofa ID 2954 from a layout specifically named 'Iago'.",
            "Find layout matches for sofa 2954 where the layout 'Dimensioni' is '23'.",
            "Are there layout suggestions for sofa 3273 in a 'small' space?",
            "Layout matches for sofa 9999 (non-existent in layouts).",
            "Layout matches for sofa 2957 if the space constraint is 'very_specific_unmatchable_constraint'.",
            "What matches the 'Iago' sofa, considering a blue color scheme?"
        ]
        # Loop through test requests and invoke the agent
        for i, manager_req in enumerate(test_requests_from_manager):
            logger.info(f"--- MatchExpert Test Case {i + 1} ---")
            logger.info(f"Manager Request to MatchExpert: \"{manager_req}\"")
            response_to_manager = match_expert.invoke(argument=manager_req, json_mode=match_expert.json_mode)
            logger.success(f"MatchExpert's Response (to Manager):\n{response_to_manager}\n")
            match_expert.reset()  # Reset agent state for next test

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}. CWD: {os.getcwd()}")
        logger.error(f"Please ensure all paths are correct. Project root was assumed to be: {project_root}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)