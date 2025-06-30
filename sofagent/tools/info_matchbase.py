import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
import re # Regular expression module
import random # For sampling results

from sofagent.tools.base import Tool


class InfoMatchbase(Tool):
    """
    Tool for finding and suggesting harmonious pairings between furniture items.
    It uses predefined collections, moodboards, and layouts to provide matching suggestions.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the InfoMatchbase tool by loading necessary CSV data files
        (matches, sofa info, furniture info, layouts, and moodboards).
        """
        super().__init__(*args, **kwargs)
        # Get file paths from the configuration
        match_info_path = self.config.get('match_info', None)
        sofa_info_path = self.config.get('sofa_info', None)
        furn_info_path = self.config.get('furn_info', None)
        layouts_path = self.config.get('layouts_info', None)
        moodboards_path = self.config.get('moodboards_info', None)

        # Load CSV files into pandas DataFrames
        self._match_info = self._load_csv(match_info_path,
                                          required_cols={'Codice_articolo': str, 'Cod_abbinamento': str, 'Tipo': str})
        self._sofa_info = self._load_csv(sofa_info_path, required_cols={'Codice': str, 'Nome': str})
        self._furn_info = self._load_csv(furn_info_path, required_cols={'article': str, 'cat2': str, 'name': str})

        self._layouts = self._load_csv(layouts_path, required_cols={'ID': int, 'Nome': str, 'Dimensioni': object})
        if self._layouts is not None: # Ensure layout item columns are strings
            for col in ['Divano1', 'Divano2', 'Poltrona1', 'Poltrona2']:
                if col in self._layouts.columns:
                    self._layouts[col] = self._layouts[col].astype(str).str.strip()

        self._moodboards = self._load_csv(moodboards_path,
                                          required_cols={'ID_abbinamento': int, 'Colore_abbinamento': str, 'Tipo': str,
                                                         'ID': str, 'Nome': str})
        if self._moodboards is not None: # Ensure moodboard item IDs are strings
            self._moodboards['ID'] = self._moodboards['ID'].astype(str)

    def _load_csv(self, path: Optional[str], required_cols: Dict[str, type] = None) -> Optional[pd.DataFrame]:
        """
        Helper function to load a CSV file into a pandas DataFrame with type checking for required columns.
        Args:
            path (Optional[str]): The path to the CSV file.
            required_cols (Dict[str, type], optional): A dictionary mapping required column names
                                                       to their expected data types.
        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame, or None if loading fails.
        """
        if not path: return None
        try:
            df = pd.read_csv(path, sep=',')
            if required_cols:
                for col, col_type in required_cols.items():
                    if col not in df.columns:
                        raise ValueError(f"Required column '{col}' not found in '{path}'.")
                    # Ensure correct data types for key columns
                    if col_type == str:
                        df[col] = df[col].astype(str).str.strip()
                    elif col_type == int and not pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64') # Use Int64 for nullable integers
                    elif col_type == object: # Typically for mixed types or when exact type isn't critical for this load step
                        pass
            return df
        except FileNotFoundError:
            print(f"Warning: File not found at {path}. This tool's functionality might be limited.")
            return None
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}. This tool's functionality might be limited.")
            return None

    def reset(self, *args, **kwargs) -> None:
        """Resets any state if necessary (currently no state to reset for this tool)."""
        pass

    def list_matching_capabilities(self) -> str:
        """
        Provides a summary of the matching capabilities of this tool, including
        types of matches and example constraints.
        Returns:
            str: A string describing the tool's matching capabilities.
        """
        capabilities = ["MatchExpert can perform generic stylistic matches."]

        # Check and report color theme matching capability
        if self._moodboards is not None and not self._moodboards.empty and 'Colore_abbinamento' in self._moodboards.columns:
            unique_colors = self._moodboards['Colore_abbinamento'].dropna().astype(str).unique()
            example_colors = list(unique_colors[:3]) if len(unique_colors) > 0 else ["no specific examples found"]
            capabilities.append(f"It can suggest items based on color themes (e.g., {', '.join(example_colors)}).")
        else:
            capabilities.append("Color theme matching data (Moodboards.csv) is not fully available.")

        # Check and report layout-based matching capability
        if self._layouts is not None and not self._layouts.empty and 'Dimensioni' in self._layouts.columns:
            unique_dims = self._layouts['Dimensioni'].dropna().astype(str).unique()
            example_dims = []
            if len(unique_dims) > 0:
                numeric_dims = [d for d in unique_dims if d.replace('.', '', 1).isdigit()]
                string_dims = [d for d in unique_dims if not d.replace('.', '', 1).isdigit()]
                example_dims.extend(numeric_dims[:2])
                example_dims.extend(string_dims[:2])
                example_dims = list(set(example_dims))[:3] # Get a few unique examples
                if not example_dims: example_dims = ["e.g., '15', 'large room'"]
            else:
                example_dims = ["no specific examples found"]
            capabilities.append(
                f"It can suggest items based on predefined layouts considering space constraints (example 'Dimensioni' values from layouts: {', '.join(example_dims)}).")
        else:
            capabilities.append("Layout-based matching data (Layouts.csv) is not fully available.")

        capabilities.append(
            "For any match type, you need to provide the category ('sofa' or 'furn') and the ID of the primary item.")
        return "\n".join(capabilities)

    def _get_item_details(self, item_id: str, item_type: str) -> Dict[str, Optional[str]]:
        """
        Helper function to retrieve basic details (name, category) for a given item ID and type.
        Args:
            item_id (str): The ID of the item.
            item_type (str): The type of the item ('sofa' or 'furn').
        Returns:
            Dict[str, Optional[str]]: A dictionary with item details.
        """
        item_id_str = str(item_id).strip()
        details = {"id": item_id_str, "type": item_type, "name": "Unknown", "category": None}
        if item_type.lower() == "sofa" and self._sofa_info is not None:
            sofa_match = self._sofa_info[self._sofa_info['Codice'] == item_id_str]
            if not sofa_match.empty: details["name"] = sofa_match['Nome'].iloc[0]
        elif item_type.lower() == "furn" and self._furn_info is not None:
            furn_match = self._furn_info[self._furn_info['article'] == item_id_str]
            if not furn_match.empty:
                details["name"] = furn_match['name'].iloc[0]
                details["category"] = furn_match['cat2'].iloc[0] # Get furniture category
        return details

    def get_matches_from_abbinamenti(self, input_category: str, input_id: str) -> List[Dict[str, Any]]:
        """
        Finds generic stylistic matches for an item based on predefined collections (Abbinamenti.csv).
        Args:
            input_category (str): The category of the input item ('sofa' or 'furn').
            input_id (str): The ID of the input item.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matched item with details.
        """
        if self._match_info is None: return [{"error": "Generic matching data (Abbinamenti.csv) not loaded."}]
        input_id_str = str(input_id).strip()
        tipo_value = "SOFA" if input_category.lower() == "sofa" else "FURN" # Determine type for lookup

        # Find all rows in Abbinamenti.csv that contain the input item
        input_item_rows = self._match_info[
            (self._match_info['Codice_articolo'] == input_id_str) & (self._match_info['Tipo'] == tipo_value)]
        if input_item_rows.empty: return [
            {"error": f"Input item {input_category} ID {input_id_str} not found in Abbinamenti."}]

        found_matches = []
        # Iterate through unique collection codes (Cod_abbinamento) associated with the input item
        for cod_abbinamento in input_item_rows['Cod_abbinamento'].unique():
            if pd.isna(cod_abbinamento): continue # Skip if collection code is NaN
            # Find all other items in the same collection, excluding the input item itself
            matching_items_df = self._match_info[
                (self._match_info['Cod_abbinamento'] == cod_abbinamento) &
                ~((self._match_info['Codice_articolo'] == input_id_str) & (self._match_info['Tipo'] == tipo_value))
                ]
            for _, row in matching_items_df.iterrows():
                match_details = self._get_item_details(row['Codice_articolo'], row['Tipo'])
                match_details['motivation'] = f"General stylistic match from collection '{cod_abbinamento}'."
                found_matches.append(match_details)
        if len(found_matches) > 10: # Limit the number of returned matches
            found_matches = random.sample(found_matches, 10)
        return found_matches if found_matches else [
            {"info": f"No distinct generic matches for {input_category} ID {input_id_str}."}]

    def get_matches_from_moodboard(self, input_category: str, input_id: str,
                                   preferred_color_theme: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Finds matches for an item based on color themes defined in moodboards (Moodboards.csv).
        Args:
            input_category (str): The category of the input item ('sofa' or 'furn').
            input_id (str): The ID of the input item.
            preferred_color_theme (Optional[str]): An optional color theme to filter by.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matched item.
        """
        if self._moodboards is None: return [{"error": "Moodboard data (Moodboards.csv) not loaded."}]
        input_id_str = str(input_id).strip()
        tipo_value = "sofa" if input_category.lower() == "sofa" else "furn"

        # Find moodboards containing the input item
        input_item_moodboards = self._moodboards[
            (self._moodboards['ID'] == input_id_str) & (self._moodboards['Tipo'].str.lower() == tipo_value)]
        if input_item_moodboards.empty: return [
            {"error": f"Input item {input_category} ID {input_id_str} not found in Moodboards."}]

        found_matches = []
        processed_moodboard_ids = set() # To avoid processing the same moodboard group multiple times

        for _, item_moodboard_row in input_item_moodboards.iterrows():
            moodboard_id = item_moodboard_row['ID_abbinamento'] # The ID of the moodboard collection
            current_moodboard_color = str(item_moodboard_row['Colore_abbinamento']).lower()

            if moodboard_id in processed_moodboard_ids: continue

            # If a preferred theme is specified, only consider moodboards matching that theme
            if preferred_color_theme and preferred_color_theme.lower() not in current_moodboard_color:
                continue

            # Find all other items in the same moodboard, excluding the input item
            moodboard_companions = self._moodboards[
                (self._moodboards['ID_abbinamento'] == moodboard_id) &
                ~((self._moodboards['ID'] == input_id_str) & (self._moodboards['Tipo'].str.lower() == tipo_value))
                ]
            for _, companion_row in moodboard_companions.iterrows():
                match_details = self._get_item_details(companion_row['ID'], companion_row['Tipo'])
                match_details['motivation'] = (
                    f"Color compatibility from moodboard '{moodboard_id}' "
                    f"(Theme: {item_moodboard_row['Colore_abbinamento']}). "
                    f"Item covering: {companion_row.get('Rivestimento', 'N/A')}.")
                # Add if not already found (to avoid duplicates if item is in multiple relevant moodboards)
                if not any(fm['id'] == match_details['id'] and fm['type'] == match_details['type'] for fm in found_matches):
                    found_matches.append(match_details)
            processed_moodboard_ids.add(moodboard_id)

        if len(found_matches) > 5: # Limit results
            found_matches = random.sample(found_matches, 5)
        return found_matches if found_matches else [{
            "info": f"No distinct color matches for {input_category} ID {input_id_str} (theme: {preferred_color_theme or 'any'})."}]

    def get_matches_from_layout(self, input_category: str, input_id: str, space_constraint: Optional[str] = None) -> \
    List[Dict[str, Any]]:
        """
        Finds matches for an item based on predefined room layouts (Layouts.csv),
        optionally filtered by a space constraint.
        Args:
            input_category (str): The category of the input item ('sofa' or 'furn').
            input_id (str): The ID of the input item.
            space_constraint (Optional[str]): A string describing the space (e.g., "15", "small room").
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matched item from the layouts.
        """
        if self._layouts is None: return [{"error": "Layout data (Layouts.csv) not loaded."}]
        input_id_str = str(input_id).strip()
        relevant_layouts = pd.DataFrame() # To store layouts containing the input item
        layout_item_cols = ['Divano1', 'Divano2', 'Poltrona1', 'Poltrona2'] # Columns that might contain the item ID

        # Find layouts that include the input item ID in any of the specified item columns
        for col in layout_item_cols:
            if col in self._layouts.columns:
                # Check if the column starts with the item_id (handles cases like "ID:config")
                condition = self._layouts[col].str.startswith(input_id_str, na=False)
                relevant_layouts = pd.concat([relevant_layouts, self._layouts[condition]])

        relevant_layouts = relevant_layouts.drop_duplicates(subset=['ID']) # Keep unique layouts
        if relevant_layouts.empty: return [
            {"error": f"Input item {input_category} ID {input_id_str} not found in any layout."}]

        # Apply space constraint if provided
        if space_constraint and 'Dimensioni' in relevant_layouts.columns:
            try:
                # Attempt to make 'Dimensioni' numeric for comparison
                relevant_layouts['Dimensioni_Numeric'] = pd.to_numeric(relevant_layouts['Dimensioni'], errors='coerce')
                sc_numeric_match = re.search(r'\d+', space_constraint) # Extract number from constraint string

                if sc_numeric_match:
                    sc_val = int(sc_numeric_match.group(0))
                    # Simple logic for size filtering
                    if "small" in space_constraint.lower():
                        relevant_layouts = relevant_layouts[relevant_layouts['Dimensioni_Numeric'] < 20]
                    elif "large" in space_constraint.lower():
                        relevant_layouts = relevant_layouts[relevant_layouts['Dimensioni_Numeric'] >= 20]
                    else: # Assume number is an upper bound
                        relevant_layouts = relevant_layouts[relevant_layouts['Dimensioni_Numeric'] <= sc_val]
                else: # Fallback for non-numeric like "large", "small"
                    if "small" in space_constraint.lower():
                        relevant_layouts = relevant_layouts[relevant_layouts['Dimensioni_Numeric'] < 20]
                    elif "large" in space_constraint.lower():
                        relevant_layouts = relevant_layouts[relevant_layouts['Dimensioni_Numeric'] >= 20]
                relevant_layouts = relevant_layouts.drop(columns=['Dimensioni_Numeric'], errors='ignore')
            except Exception as e:
                print(f"Warning: Could not apply space constraint '{space_constraint}': {e}")

        if relevant_layouts.empty: return [
            {"info": f"No layouts for {input_id_str} match space constraint '{space_constraint}'."}]

        found_matches = []
        for _, layout_row in relevant_layouts.iterrows():
            layout_name = layout_row.get('Nome', f"Layout ID {layout_row['ID']}")
            layout_dim = layout_row.get('Dimensioni', 'N/A')
            # Iterate through item columns in the layout
            for col_name in layout_item_cols:
                if col_name in layout_row and pd.notna(layout_row[col_name]):
                    item_val_in_layout = str(layout_row[col_name])
                    current_item_id = item_val_in_layout.split(':')[0].strip() # Extract ID part
                    if current_item_id == input_id_str: continue # Don't match the input item with itself

                    # Guess item type based on column name (can be refined)
                    item_type_guess = "sofa" if "Divano" in col_name else "furn"
                    match_details = self._get_item_details(current_item_id, item_type_guess)

                    match_details['motivation'] = f"Found in layout '{layout_name}' (Dim: {layout_dim} sqm). Item role: {col_name}."
                    match_details['original_layout_role'] = col_name # Store its role in the layout
                    match_details['original_layout_config'] = item_val_in_layout # Store full item string from layout
                    # Add if not already found from this specific layout role to avoid duplicates from same layout item
                    if not any(fm['id'] == match_details['id'] and fm.get('original_layout_role') == col_name for fm in found_matches):
                        found_matches.append(match_details)

        if len(found_matches) > 5: # Limit results
            found_matches = random.sample(found_matches, 5)

        return found_matches if found_matches else [
            {"info": f"No other items in layouts with {input_id_str} (constraint: {space_constraint or 'any'})."}]