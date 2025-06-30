import pandas as pd
import re
from typing import Union, List, Dict, Any, Tuple
from difflib import get_close_matches  # For fuzzy matching

from sofagent.tools.base import Tool


class InfoFurnbase(Tool):
    """
    Tool for accessing and querying information about various non-sofa furniture items.
    It loads data from multiple CSV files, each corresponding to a furniture category.
    """
    def __init__(self, *args, **kwargs) -> None:
        '''
        Initialize the InfoFurnbase tool by loading all furniture category catalogs
        based on the provided configuration.
        '''
        super().__init__(*args, **kwargs)
        self.category_dataframes: Dict[str, pd.DataFrame] = {} # Stores DataFrames for each category
        self.category_id_column: Dict[str, str] = {}       # Maps category to its specific ID column name
        self.category_name_column: Dict[str, str] = {}     # Maps category to its specific name column name
        self.category_price_column: Dict[str, str] = {}    # Maps category to its specific price column name

        # Configuration mapping category names to their data file keys and relevant column names
        category_configs = {
            "ArrediVari": {"path_key": "other_info", "id_col": "codice_articolo", "name_col": "nome_modello_o_prodotto",
                           "price_col": "prezzo"},
            "ArteParete": {"path_key": "wallarts_info", "id_col": "codice_articolo", "name_col": "titolo_o_serie",
                           "price_col": "prezzo"},
            "Cassettiere": {"path_key": "dressers_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                            "price_col": "prezzo"},
            "Comodini": {"path_key": "nightstands_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                         "price_col": "prezzo"},
            "CredenzeMobiliContenitori": {"path_key": "sideboards_info", "id_col": "codice_articolo",
                                          "name_col": "nome_modello", "price_col": "prezzo"},
            "Lampade": {"path_key": "lamps_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                        "price_col": "prezzo"},
            "Librerie": {"path_key": "libraries_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                         "price_col": "prezzo"},
            "MaterassiGuanciali": {"path_key": "matresses_info", "id_col": "codice_articolo",
                                   "name_col": "nome_modello", "price_col": "prezzo"},
            "OggettiDecorativi": {"path_key": "decorations_info", "id_col": "codice_articolo",
                                  "name_col": "nome_modello_o_linea", "price_col": "prezzo"},
            "Profumatori": {"path_key": "fragrances_info", "id_col": "codice_articolo",
                            "name_col": "nome_linea_o_fragranza", "price_col": "prezzo"},
            "SediePoltroncine": {"path_key": "chairs_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                                 "price_col": "prezzo"},
            "Specchi": {"path_key": "mirrors_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                        "price_col": "prezzo"},
            "Tappeti": {"path_key": "rugs_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                        "price_col": "prezzo"},
            "TavoliniCaffe": {"path_key": "coffe_tables_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                              "price_col": "prezzo"},
            "TavoliPranzo": {"path_key": "dining_tables_info", "id_col": "codice_articolo", "name_col": "nome_modello",
                             "price_col": "prezzo"},
            "Tessili": {"path_key": "textiles_info", "id_col": "codice_articolo", "name_col": "nome_linea_modello",
                        "price_col": "prezzo"},
        }
        self.valid_category_names_original_case = list(category_configs.keys()) # Store original casing for user messages

        # Load data for each configured category
        for cat_name, config_details in category_configs.items():
            file_path = self.config.get(config_details["path_key"], None)
            if file_path:
                try:
                    df = pd.read_csv(file_path, sep=',')
                    id_col, name_col, price_col = config_details["id_col"], config_details["name_col"], config_details[
                        "price_col"]

                    # Ensure required columns exist and have correct types
                    if id_col not in df.columns:
                        print(f"Warning: ID column '{id_col}' not found in {cat_name}. Skipping."); continue
                    df[id_col] = df[id_col].astype(str) # ID is always string

                    if name_col in df.columns:
                        df[name_col] = df[name_col].astype(str) # Name is string
                    else:
                        print(f"Warning: Name column '{name_col}' not found in {cat_name}.")

                    if price_col in df.columns:
                        df[price_col] = pd.to_numeric(df[price_col], errors='coerce') # Price should be numeric
                    else:
                        print(f"Warning: Price column '{price_col}' not found in {cat_name}.")

                    # Store DataFrames and column names using lowercase category names as keys for consistency
                    self.category_dataframes[cat_name.lower()] = df
                    self.category_id_column[cat_name.lower()] = id_col
                    self.category_name_column[cat_name.lower()] = name_col
                    self.category_price_column[cat_name.lower()] = price_col
                except Exception as e:
                    print(f"Warning: Error loading category {cat_name} from {file_path}: {e}")
            else:
                print(f"Warning: Path not configured for category {cat_name}")
        if not self.category_dataframes: raise ValueError("No furniture category dataframes loaded.")

    def reset(self, *args, **kwargs) -> None:
        """Resets any state if necessary (currently no state to reset for this tool)."""
        pass

    def _get_category_df_and_cols(self, category: str) -> Union[tuple[pd.DataFrame, str, str, str, str], str]:
        """
        Helper function to retrieve the DataFrame and relevant column names for a given category.
        It handles case-insensitivity and fuzzy matching for category names.
        Args:
            category (str): The category name to look up.
        Returns:
            Union[tuple[pd.DataFrame, str, str, str, str], str]: A tuple containing the DataFrame,
            ID column, name column, price column, and matched category name (original case),
            or an error string if the category is not found.
        """
        cat_lower = category.lower().strip()
        if cat_lower in self.category_dataframes: # Direct match (case-insensitive)
            original_case_cat_name = next(
                (c for c in self.valid_category_names_original_case if c.lower() == cat_lower), category)
            return (self.category_dataframes[cat_lower],
                    self.category_id_column[cat_lower],
                    self.category_name_column.get(cat_lower),
                    self.category_price_column.get(cat_lower),
                    original_case_cat_name)
        # Attempt fuzzy matching if direct match fails
        close_matches = get_close_matches(cat_lower, self.category_dataframes.keys(), n=1, cutoff=0.7)
        if close_matches:
            matched_key_lower = close_matches[0]
            original_case_cat_name = next(
                (c for c in self.valid_category_names_original_case if c.lower() == matched_key_lower),
                matched_key_lower) # Get original casing for user-facing messages
            return (self.category_dataframes[matched_key_lower],
                    self.category_id_column[matched_key_lower],
                    self.category_name_column.get(matched_key_lower),
                    self.category_price_column.get(matched_key_lower),
                    original_case_cat_name)
        error_msg = (f"Error: Category '{category}' not recognized. "
                     f"Valid categories are: {', '.join(self.valid_category_names_original_case)}")
        return error_msg

    def get_furn_id_by_name(self, category: str, furn_name: str) -> str:
        """
        Retrieves the ID of a furniture item given its category and name.
        Args:
            category (str): The furniture category.
            furn_name (str): The name of the furniture item.
        Returns:
            str: A message containing the item's ID or an error message.
        """
        cat_data_or_error = self._get_category_df_and_cols(category)
        if isinstance(cat_data_or_error, str): return cat_data_or_error
        df, id_col, name_col, _, matched_cat_name = cat_data_or_error

        if not name_col or name_col not in df.columns:
            return f"Error: Name column not configured or found for category '{matched_cat_name}'."

        furn_name_clean = str(furn_name).strip().lower()
        # Exact match on cleaned name (case-insensitive)
        matches = df[df[name_col].str.lower() == furn_name_clean]

        if not matches.empty:
            ids = matches[id_col].unique()
            if len(ids) == 1:
                return f"The ID for item '{furn_name}' in category '{matched_cat_name}' is {ids[0]}."
            else: # Multiple items with the same name
                return f"Multiple items found with name '{furn_name}' in category '{matched_cat_name}'. IDs: {', '.join(ids)}. Please specify one."
        else:
            return f"Error: Item with name '{furn_name}' not found in category '{matched_cat_name}'."


    def furn_info_and_price(self, category_items: List[List[Union[str, int]]]) -> str:
        """
        Retrieves detailed information and price for one or more furniture items,
        given their category and ID or name.
        Args:
            category_items (List[List[Union[str, int]]]): A list of pairs, where each pair is
                                                         [category_name, item_id_or_name].
        Returns:
            str: A formatted string with information for each found item, or error messages.
        """
        results_list = []
        if not isinstance(category_items, list):
            return "Invalid input. Expected a list of [Category, ID_or_Name] pairs."

        for item_pair in category_items:
            if not (isinstance(item_pair, list) and len(item_pair) == 2):
                results_list.append(f"Invalid item query format: {item_pair}. Expected [Category, ID_or_Name].")
                continue
            category, item_query = str(item_pair[0]), str(item_pair[1]).strip()
            cat_data_or_error = self._get_category_df_and_cols(category)
            if isinstance(cat_data_or_error, str): # Category not found or error
                results_list.append(f"For item '{item_query}': {cat_data_or_error}")
                continue

            df, id_col, name_col, price_col, matched_cat_name = cat_data_or_error
            item_data = pd.DataFrame() # To store found item data

            # Try to find by ID first (exact match, case-insensitive)
            if id_col in df.columns:
                item_data = df[df[id_col].str.fullmatch(item_query, case=False, na=False)]
            # If not found by ID, try by name (contains, case-insensitive)
            if item_data.empty and name_col and name_col in df.columns:
                item_data = df[df[name_col].str.contains(item_query, case=False, na=False)]

            if not item_data.empty:
                for _, row in item_data.iterrows():
                    features = {k: v for k, v in row.to_dict().items() if pd.notna(v)} # Get all non-null features
                    desc = [f"Category: {matched_cat_name}"]
                    item_ident_str = features.get(id_col, item_query) # Use actual ID if found, else the query
                    desc.append(f"ID: {item_ident_str}")
                    if name_col and name_col in features and name_col != id_col: # Add name if available and different from ID
                        desc.append(f"Name: {features[name_col]}")

                    # Add other relevant features, excluding already mentioned and internal ones
                    other_features = {k: v for k, v in features.items() if
                                      k not in [id_col, name_col, 'Category', price_col]}
                    for k, v_val in other_features.items(): desc.append(f"{k.replace('_', ' ').capitalize()}: {v_val}")

                    price_val = features.get(price_col, "N/A")
                    desc.append(f"Price: {price_val} EUR" if price_val != "N/A" else "Price: N/A")
                    results_list.append(f"Item Found: {'; '.join(desc)}")
            else:
                results_list.append(f"Item '{item_query}' not found in category '{matched_cat_name}'.")

        return "\n".join(results_list) if results_list else "No items processed or found."

    def list_furn_features_for_category(self, category: str) -> str:
        """
        Lists the queryable features for a given furniture category, along with example values.
        Args:
            category (str): The furniture category name.
        Returns:
            str: A formatted string listing queryable features or an error message.
        """
        cat_data_or_error = self._get_category_df_and_cols(category)
        if isinstance(cat_data_or_error, str):
            return cat_data_or_error
        df, id_col, name_col, price_col, matched_cat_name = cat_data_or_error

        output_lines = [f"Queryable features for category '{matched_cat_name}':"]
        # Columns to generally exclude from user-facing queryable features
        excluded_cols = ['quantita', 'marca', 'categoria_prodotto', 'codice_stato', 'codice_articolo_alternativo', id_col]
        if name_col: excluded_cols.append(name_col)
        if price_col: excluded_cols.append(price_col)

        for col in df.columns:
            if col in excluded_cols:
                continue

            example_values = []
            unique_vals = df[col].dropna().astype(str).unique() # Get unique non-null values

            if len(unique_vals) > 0:
                sorted_unique_vals = sorted(list(unique_vals), key=len) # Prioritize shorter examples
                example_values = [str(v) for v in sorted_unique_vals[:3]]
                example_str = ", ".join(example_values)
                output_lines.append(f"- **{col}** (e.g., {example_str})") # Add column name and examples
            else:
                output_lines.append(f"- **{col}** (no distinct examples found in data)")

        if price_col: # Add specific mention for the price column
            output_lines.append(
            f"- **{price_col}** (numeric, e.g., {df[price_col].min()}-{df[price_col].max() if pd.notna(df[price_col].max()) else 'N/A'}) - Use for price conditions.")
        output_lines.append("\nUse 'FeatureName=Value' or 'FeatureName CONTAINS Value'. Combine with '&&'.")
        return "\n".join(output_lines)

    def _parse_conditions(self, condition_str: str, df: pd.DataFrame, category_name_for_parser: str) -> Tuple[
        pd.Series, str]:
        """
        Parses a string of combined conditions (ANDed) and applies them to a DataFrame.
        Args:
            condition_str (str): The condition string (e.g., "feature1=value1 && feature2 CONTAINS value2").
            df (pd.DataFrame): The DataFrame to filter.
            category_name_for_parser (str): The (lowercase) category name, used to fetch the correct price column.
        Returns:
            Tuple[pd.Series, str]: A boolean Series for filtering and an error message string (empty if no error).
        """
        conditions = condition_str.split('&&')
        combined_filter = pd.Series([True] * len(df), index=df.index) # Start with all True
        price_col_for_cat = self.category_price_column.get(category_name_for_parser.lower()) # Get price col for this category

        for cond_part in conditions:
            cond_part = cond_part.strip()
            # Regex to extract column_name, operator, and value
            match = re.match(r"^\s*([\w\s.-]+)\s*([!=<>]=?|CONTAINS)\s*(['\"]?.*['\"]?)\s*$", cond_part, re.IGNORECASE)
            if not match: return pd.Series([False] * len(df),
                                           index=df.index), f"Invalid condition format: '{cond_part}'"

            column_name, operator, value_str = match.groups()
            column_name, operator, value_str = column_name.strip(), operator.strip().upper(), value_str.strip().strip("'\"")

            # Find actual column name in DataFrame (case-insensitive)
            actual_column_name = next((col for col in df.columns if col.lower() == column_name.lower()), None)
            if not actual_column_name: return pd.Series([False] * len(df),
                                                        index=df.index), f"Error: Feature '{column_name}' not found for this category."

            target_column_series = df[actual_column_name]
            try:
                current_filter = pd.Series([False] * len(df), index=df.index)
                # Determine if it's a numeric comparison (based on dtype, operator, or if it's the price column)
                is_numeric_comparison = (pd.api.types.is_numeric_dtype(
                    target_column_series.dropna().dtype) and target_column_series.dropna().size > 0) or \
                                        operator in ['<', '>', '<=', '>='] or \
                                        (price_col_for_cat and actual_column_name.lower() == price_col_for_cat.lower())

                if is_numeric_comparison:
                    try:
                        value_num = float(value_str)
                    except ValueError:
                        return pd.Series([False] * len(df),
                                         index=df.index), f"Error: Value '{value_str}' for feature '{actual_column_name}' must be a number for operator '{operator}'."
                    target_numeric = pd.to_numeric(target_column_series, errors='coerce') # Convert target column to numeric
                    # Apply numeric operator
                    if operator == '=' or operator == '==': current_filter = (target_numeric == value_num)
                    elif operator == '!=': current_filter = (target_numeric != value_num)
                    elif operator == '<': current_filter = (target_numeric < value_num)
                    elif operator == '>': current_filter = (target_numeric > value_num)
                    elif operator == '<=': current_filter = (target_numeric <= value_num)
                    elif operator == '>=': current_filter = (target_numeric >= value_num)
                    else: return pd.Series([False] * len(df), index=df.index), f"Error: Operator '{operator}' not valid for numeric feature '{actual_column_name}'."
                    current_filter = current_filter.fillna(False) # Treat NaN comparisons as False
                else:  # String operations
                    target_str_series = target_column_series.astype(str).str.lower() # Convert to lowercase string
                    value_str_lower = value_str.lower()
                    if operator == '=' or operator == '==': current_filter = (target_str_series == value_str_lower)
                    elif operator == '!=': current_filter = (target_str_series != value_str_lower)
                    elif operator == 'CONTAINS': current_filter = target_str_series.str.contains(value_str_lower, na=False)
                    else: return pd.Series([False] * len(df), index=df.index), f"Error: Operator '{operator}' not valid for text feature '{actual_column_name}'. Use '=', '!=', or 'CONTAINS'."
                combined_filter &= current_filter # AND with previous conditions
            except Exception as e:
                return pd.Series([False] * len(df),
                                 index=df.index), f"Error applying condition '{cond_part}' on '{actual_column_name}': {e}"
        return combined_filter, "" # Return combined filter and empty error message

    def _format_furn_search_results(self, df_results: pd.DataFrame, category: str, id_col: str, name_col: str,
                                    price_col: str = None, N: int = 5) -> str:
        """
        Formats a DataFrame of furniture search results into a string for the agent.
        Args:
            df_results (pd.DataFrame): DataFrame containing the items that matched the query.
            category (str): The category of the items.
            id_col (str): The name of the ID column for this category.
            name_col (str): The name of the name column for this category.
            price_col (str, optional): The name of the price column. Defaults to None.
            N (int, optional): The maximum number of random results to display. Defaults to 5.
        Returns:
            str: A formatted string of search results.
        """
        if df_results.empty: return "No items found matching the criteria."
        results_output = []
        sample_n = min(N, len(df_results)) # Show at most N items
        # Sample randomly if more than N results, otherwise show all
        df_to_display = df_results.sample(n=sample_n, random_state=42) if len(
            df_results) > sample_n else df_results.sample(frac=1, random_state=42).head(sample_n)

        for _, row in df_to_display.iterrows():
            item_id = row.get(id_col, "N/A")
            item_name_val = row.get(name_col)
            item_name = f" ({item_name_val})" if name_col and pd.notna(item_name_val) and item_name_val else ""
            price_info = ""
            if price_col and price_col in row and pd.notna(row[price_col]): # Add price if available
                price_info = f" - Price: {row[price_col]} EUR"
            results_output.append(f"Category: {category}, ID: {item_id}{item_name}{price_info}")

        num_found = len(df_results)
        final_str = f"Found {num_found} item(s) in category '{category}'. "
        final_str += f"Showing {sample_n} random results:\n" if num_found > N else "Details:\n" if num_found > 0 else ""
        final_str += "\n".join(results_output)
        return final_str

    def furn_feature_query(self, category: str, condition_str: str) -> str:
        """
        Queries furniture items based on feature conditions within a specific category.
        Args:
            category (str): The furniture category to search within.
            condition_str (str): The feature-based condition string (e.g., "materiale_struttura=Metallo").
        Returns:
            str: Formatted search results or an error message.
        """
        cat_data_or_error = self._get_category_df_and_cols(category)
        if isinstance(cat_data_or_error, str): return cat_data_or_error # Return error if category invalid
        df, id_col, name_col, price_col, matched_cat_name = cat_data_or_error

        # Handle empty condition string (interpret as request for random items)
        if not condition_str or not condition_str.strip() or condition_str.strip() in ["''", '""']:
            sample_size = min(5, len(df))
            if sample_size == 0: return f"No items in '{matched_cat_name}' to sample."
            random_items = df.sample(n=sample_size, random_state=42)
            return f"No criteria for '{matched_cat_name}'. Here are 5 random items:\n" + \
                self._format_furn_search_results(random_items, matched_cat_name, id_col, name_col, price_col, N=sample_size)

        df.attrs['category_name_for_parser'] = matched_cat_name.lower() # Pass category to parser for context
        filter_series, error_msg = self._parse_conditions(condition_str, df, matched_cat_name)
        if error_msg: return f"Error in '{matched_cat_name}': {error_msg}"

        matching_items = df[filter_series]
        return self._format_furn_search_results(matching_items, matched_cat_name, id_col, name_col, price_col, N=5)

    def furn_price_query(self, category: str, price_condition_str: str) -> str:
        """
        Queries furniture items based on price conditions within a specific category.
        Args:
            category (str): The furniture category.
            price_condition_str (str): The price-based condition string (e.g., "prezzo < 500").
        Returns:
            str: Formatted search results or an error message.
        """
        cat_data_or_error = self._get_category_df_and_cols(category)
        if isinstance(cat_data_or_error, str): return cat_data_or_error
        df, id_col, name_col, price_col, matched_cat_name = cat_data_or_error

        if not price_col or price_col not in df.columns: # Ensure price column exists for this category
            return f"Error: Price column ('{price_col if price_col else 'undefined'}') not found for category '{matched_cat_name}'."

        df.attrs['category_name_for_parser'] = matched_cat_name.lower()
        df_priced = df.dropna(subset=[price_col]) # Only consider items with price information
        if df_priced.empty: return f"No items with price info in '{matched_cat_name}'."

        # Handle empty price condition (show random priced items)
        if not price_condition_str or not price_condition_str.strip() or price_condition_str.strip() in ["''", '""']:
            sample_size = min(5, len(df_priced))
            if sample_size == 0: return f"No items with price in '{matched_cat_name}' to sample."
            random_items = df_priced.sample(n=sample_size, random_state=42)
            return f"No price criteria for '{matched_cat_name}'. Here are 5 random priced items:\n" + \
                self._format_furn_search_results(random_items, matched_cat_name, id_col, name_col, price_col, N=sample_size)

        filter_series, error_msg = self._parse_conditions(price_condition_str, df_priced, matched_cat_name)
        if error_msg: return f"Error in '{matched_cat_name}': {error_msg}"

        matching_items = df_priced[filter_series]
        return self._format_furn_search_results(matching_items, matched_cat_name, id_col, name_col, price_col, N=5)

    def furn_combo_query(self, category: str, complex_condition_str: str) -> str:
        """
        Queries furniture items based on a combination of feature and price conditions.
        Args:
            category (str): The furniture category.
            complex_condition_str (str): The combined condition string (e.g., "materiale=Legno && prezzo > 1000").
        Returns:
            str: Formatted search results or an error message.
        """
        cat_data_or_error = self._get_category_df_and_cols(category)
        if isinstance(cat_data_or_error, str): return cat_data_or_error
        df, id_col, name_col, price_col, matched_cat_name = cat_data_or_error

        if not price_col or price_col not in df.columns: # Price column is essential for combo queries involving price
            return f"Error: Price column ('{price_col if price_col else 'undefined'}') not configured or found for combo query in '{matched_cat_name}'."

        df.attrs['category_name_for_parser'] = matched_cat_name.lower()

        # Handle empty complex condition (show random priced items from the category)
        if not complex_condition_str or not complex_condition_str.strip() or complex_condition_str.strip() in ["''", '""']:
            df_priced = df.dropna(subset=[price_col])
            sample_size = min(5, len(df_priced))
            if sample_size == 0: return f"No items with price in '{matched_cat_name}' to sample for empty combo."
            random_items = df_priced.sample(n=sample_size, random_state=42)
            return f"No criteria for combo query in '{matched_cat_name}'. Random sample of 5 items:\n" + \
                self._format_furn_search_results(random_items, matched_cat_name, id_col, name_col, price_col, N=sample_size)

        # Separate feature conditions from price conditions
        all_conditions = complex_condition_str.split('&&')
        feature_conditions_list, price_conditions_list = [], []
        for cond in all_conditions:
            cond_strip = cond.strip()
            col_match = re.match(r"^\s*([\w\s.-]+)\s*([!=<>]=?|CONTAINS)\s*(.+)\s*$", cond_strip, re.IGNORECASE)
            is_price_cond = False
            if col_match: # Check if the condition column is the designated price column
                col_name_from_cond = col_match.groups()[0].strip().lower()
                if price_col and col_name_from_cond == price_col.lower(): is_price_cond = True
            # Broader check: if it uses price-related keywords or numeric comparison operators
            if not is_price_cond and (re.search(r'\b(?:prezzo|price|sell_out)\b', cond_strip, re.IGNORECASE) or \
                                      (col_match and col_match.groups()[1].strip() in ['<', '>', '<=', '>=', '==', '!='] and \
                                       col_match.groups()[2].strip().replace('.', '', 1).lstrip('-+').isdigit())):
                is_price_cond = True
            if is_price_cond:
                price_conditions_list.append(cond_strip)
            else:
                feature_conditions_list.append(cond_strip)

        current_df_results = df.copy()
        # Apply feature conditions first
        if feature_conditions_list:
            feat_cond_str = " && ".join(feature_conditions_list)
            feat_filter, err_msg = self._parse_conditions(feat_cond_str, current_df_results, matched_cat_name)
            if err_msg: return f"Error parsing feature conditions for '{matched_cat_name}': {err_msg}"
            current_df_results = current_df_results[feat_filter]
        if current_df_results.empty:
            return f"No items in '{matched_cat_name}' found matching features: {' && '.join(feature_conditions_list) or 'N/A'}"

        # Then, apply price conditions to the already feature-filtered results
        if price_conditions_list and price_col and price_col in current_df_results.columns:
            current_df_results[price_col] = pd.to_numeric(current_df_results[price_col], errors='coerce')
            current_df_results = current_df_results.dropna(subset=[price_col]) # Remove items without valid price after feature filtering
            if current_df_results.empty:
                return f"Items in '{matched_cat_name}' matched features, but have no valid price data for further filtering by: {' && '.join(price_conditions_list)}."

            price_cond_str = " && ".join(price_conditions_list)
            price_filter, err_msg = self._parse_conditions(price_cond_str, current_df_results, matched_cat_name)
            if err_msg: return f"Error parsing price conditions for '{matched_cat_name}': {err_msg}"
            current_df_results = current_df_results[price_filter]

        if current_df_results.empty:
            return f"No items in '{matched_cat_name}' found for combined conditions: {complex_condition_str}"
        return self._format_furn_search_results(current_df_results, matched_cat_name, id_col, name_col, price_col, N=5)