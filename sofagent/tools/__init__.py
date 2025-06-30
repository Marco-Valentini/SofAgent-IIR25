from sofagent.tools.base import Tool
from sofagent.tools.info_sofabase import InfoSofabase
from sofagent.tools.info_matchbase import InfoMatchbase
from sofagent.tools.info_furnbase import InfoFurnbase

TOOL_MAP: dict[str, type] = {
    'info_sofa': InfoSofabase,
    'info_match': InfoMatchbase,
    'info_furn': InfoFurnbase,
}