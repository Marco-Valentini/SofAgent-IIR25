import streamlit as st
from typing import Optional


def add_chat_message(role: str, message: str, avatar: Optional[str] = None):
    """Add a chat message to the chat history.

    Args:
        `role` (`str`): The role of the message.
        `message` (`str`): The message to be added.
        `avatar` (`Optional[str]`): The avatar of the agent. If `avatar` is `None`, use the default avatar.
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': role, 'message': message})
    if avatar is not None:
        st.chat_message(role, avatar=avatar).markdown(message, unsafe_allow_html=True) # Ensure unsafe_allow_html is True here too
    else:
        st.chat_message(role).markdown(message, unsafe_allow_html=True) # And here

# Modifica qui per modificare il colore del nome agente nella conversazione
def get_color_style(agent_type: str) -> str:
    """Get the CSS color style string for the agent's name.

    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The CSS color style string (e.g., 'color:#A52A2A;').
    """
    agent_type_lower = agent_type.lower()
    color_hex = ""

    if 'manager' in agent_type_lower:
        color_hex = '#A52A2A'  # A nice brown
    elif 'furnanalyzer' in agent_type_lower:  # Matches FurnAnalyzer class name
        color_hex = '#FF8C00'  # DarkOrange
    elif 'sofaanalyzer' in agent_type_lower:  # Matches SofaAnalyzer class name
        color_hex = '#E65100'  # A deeper orange/red
    elif 'furnsearcher' in agent_type_lower:  # Matches FurnSearcher
        color_hex = '#4682B4'  # SteelBlue
    elif 'sofasearcher' in agent_type_lower:  # Matches SofaSearcher
        color_hex = '#1E90FF'  # DodgerBlue
    elif 'matchexpert' in agent_type_lower:
        color_hex = '#2E8B57'  # SeaGreen
    elif 'assistant' == agent_type_lower:  # For default system messages when agent is None
        color_hex = '#556B2F'  # DarkOliveGreen
    else:  # Default fallback for other agent types or if no match
        color_hex = '#708090'  # SlateGray (a nice gray)

    return f'color:{color_hex};'

def get_avatar(agent_type: str) -> str:
    """Get the avatar of the agent.

    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The avatar of the agent.
    """
    if 'manager' in agent_type.lower():
        return '👨🏻‍💼'
    elif 'searcher' in agent_type.lower(): # Covers SofaSearcher and FurnSearcher
        return '🔍'
    elif 'analyzer' in agent_type.lower(): # Covers SofaAnalyzer and FurnAnalyzer
        return '📊' # Example, choose an appropriate one
    elif 'match' in agent_type.lower(): # Covers MatchExpert
        return '🔗' # Example
    else: # Assistant or other
        return '🤖'
