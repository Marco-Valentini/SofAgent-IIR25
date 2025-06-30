import pandas as pd
import re
from typing import Union, List, Dict, Any, Tuple

from sofagent.tools.base import Tool


class InfoSofabase(Tool):
    """
    Tool for accessing and querying information about sofas, including catalog details,
    pricing, and predefined configurations.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the InfoSofabase tool by loading necessary CSV data files
        (sofa catalog, price information, and predefined configurations with seats/prices).
        """
        super().__init__(*args, **kwargs)
        # Get file paths from the configuration passed by the agent
        sofa_catalog_path = self.config.get('sofa_info', None)
        sofa_prices_path = self.config.get('price_info', None)
        sofa_predefined_configs_path = self.config.get('price_seats_info', None)

        # Load main sofa catalog
        if sofa_catalog_path:
            self._sofa_info = pd.read_csv(sofa_catalog_path, sep=',')
            if 'Codice' in self._sofa_info.columns: # 'Codice' is the sofa ID
                self._sofa_info['Codice'] = self._sofa_info['Codice'].astype(str)
            else:
                raise ValueError("Column 'Codice' not found in sofa_info (sofa catalog).")
            if 'Nome' in self._sofa_info.columns: self._sofa_info['Nome'] = self._sofa_info['Nome'].astype(str)
        else:
            raise ValueError("sofa_catalog_path (sofa_info) is required for InfoSofabase.")

        # Load sofa price information
        if sofa_prices_path is not None:
            self._price_info = pd.read_csv(sofa_prices_path, sep=',')
            if 'Codice' in self._price_info.columns:
                self._price_info['Codice'] = self._price_info['Codice'].astype(str)
            else:
                raise ValueError("Sofa ID column ('Codice') not found in price_info.")
            if 'Sell_out' not in self._price_info.columns: # Assumed to be the price column
                raise ValueError("Column 'Sell_out' not found in price_info (PrezziDivani.csv).")
            self._price_info['Sell_out'] = pd.to_numeric(self._price_info['Sell_out'], errors='coerce')
            # Identify configuration columns (e.g., Configurazione_1, Configurazione_2, etc.)
            self.config_cols = [col for col in self._price_info.columns if col.startswith("Configurazione_")]
            for col in self.config_cols: # Normalize configuration values
                self._price_info[col] = self._price_info[col].astype(str).str.lower().str.strip()
                self._price_info[col].replace('nan', pd.NA, inplace=True) # Treat 'nan' strings as missing
        else:
            raise ValueError("sofa_prices_path (price_info / PrezziDivani.csv) is required for InfoSofabase.")

        # Load predefined sofa configurations (with seats and calculated prices)
        if sofa_predefined_configs_path:
            self._predefined_configs_info = pd.read_csv(sofa_predefined_configs_path, sep=',')
            if 'ID' in self._predefined_configs_info.columns: # 'ID' here refers to sofa Codice
                self._predefined_configs_info['ID'] = self._predefined_configs_info['ID'].astype(str)
            else: # Changed error message to reflect correct column name
                raise ValueError("Column 'ID' not found in predefined_configs_info (ConfigurazioniPredefiniteDivani_ConPrezziESedute.csv).")
            if 'Total_Seats' in self._predefined_configs_info.columns:
                self._predefined_configs_info['Total_Seats'] = pd.to_numeric(
                    self._predefined_configs_info['Total_Seats'], errors='coerce')
            # Note: 'Total_Price_EUR' is also expected and should ideally be checked/converted if necessary
        else:
            raise ValueError("sofa_predefined_configs_path (price_seats_info / ConfigurazioniPredefiniteDivani_ConPrezziESedute.csv) is required.")


    def reset(self, *args, **kwargs) -> None:
        """Resets any state if necessary (currently no state to reset for this tool)."""
        pass

    def get_sofa_id_by_name(self, sofa_name: str) -> str:
        """
        Retrieves the ID(s) of a sofa given its name.
        Args:
            sofa_name (str): The name of the sofa.
        Returns:
            str: A message containing the sofa's ID or an error/ambiguity message.
        """
        if not hasattr(self, '_sofa_info') or self._sofa_info is None:
            return "Error: Sofa catalog not loaded."
        sofa_name_clean = str(sofa_name).strip().lower()
        # Case-insensitive exact match for sofa name
        matches = self._sofa_info[self._sofa_info['Nome'].str.lower() == sofa_name_clean]
        if not matches.empty:
            ids = matches['Codice'].unique()
            if len(ids) == 1:
                return f"The ID for sofa '{sofa_name}' is {ids[0]}."
            else: # Handles cases where multiple sofas might share the same name
                return f"Multiple sofas found with name '{sofa_name}'. IDs: {', '.join(ids)}. Please specify one."
        else:
            return f"Error: Sofa with name '{sofa_name}' not found."

    def sofa_info(self, sofa_ids_or_names: Union[str, int, List[Union[str, int]]]) -> str:
        """
        Retrieves detailed information for one or more sofas, identified by ID(s) or name(s).
        Args:
            sofa_ids_or_names (Union[str, int, List[Union[str, int]]]): A single ID/name or a list of IDs/names.
        Returns:
            str: A formatted string with information for each found sofa, or error messages.
        """
        if not hasattr(self, '_sofa_info') or self._sofa_info is None:
            return "Error: Sofa catalog (sofa_info) is not loaded."

        # Standardize input to a list of strings
        if isinstance(sofa_ids_or_names, (str, int)):
            inputs = [str(sofa_ids_or_names).strip()]
        elif isinstance(sofa_ids_or_names, list):
            inputs = [str(item).strip() for item in sofa_ids_or_names]
        else:
            return "Invalid input type for sofa_ids_or_names. Expected string, int, or list."

        results_list = []
        for item_query_str in inputs:
            if not item_query_str: continue

            # Attempt to match by ID (Codice) first (exact, case-insensitive)
            sofa_data_id = self._sofa_info[self._sofa_info['Codice'].str.fullmatch(item_query_str, case=False, na=False)]
            # If not found by ID, attempt to match by Name (contains, case-insensitive)
            sofa_data_name = self._sofa_info[self._sofa_info['Nome'].str.contains(item_query_str, case=False, na=False)]

            # Combine results, prioritizing ID matches and removing duplicates
            sofa_data = pd.concat([sofa_data_id, sofa_data_name]).drop_duplicates(subset=['Codice']).reset_index(drop=True)

            if not sofa_data.empty:
                for _, row in sofa_data.iterrows():
                    features = {k: v for k, v in row.to_dict().items() if pd.notna(v)} # Get all non-null features
                    feature_strings = []
                    if 'Codice' in features: feature_strings.append(f"ID: {features['Codice']}")
                    if 'Nome' in features: feature_strings.append(f"Name: {features['Nome']}")

                    # Exclude certain columns from the "other features" list
                    other_features = {k: v for k, v in features.items() if
                                      k not in ['Codice', 'Nome', 'Descrizione'] and not k.startswith('Unnamed')}
                    for k, v_val in other_features.items():
                        feature_strings.append(f"{k.replace('_', ' ').capitalize()}: {v_val}") # Format feature name

                    if 'Descrizione' in features: # Add description if available
                        feature_strings.append(f"Description: {features['Descrizione']}")
                    results_list.append(f"Sofa Found: {'; '.join(feature_strings)}")
            else:
                results_list.append(f"Sofa with ID/Name '{item_query_str}' not found in the catalog.")
        return "\n".join(results_list) if results_list else "No sofas queried or no information found."

    def _format_price_results(self, price_df: pd.DataFrame, sofa_id_str: str = "") -> str:
        """
        Helper function to format sofa price information into a readable string.
        Args:
            price_df (pd.DataFrame): DataFrame containing price information for configurations.
            sofa_id_str (str, optional): The sofa ID, used if not present in every row of price_df.
        Returns:
            str: Formatted string of price results.
        """
        if price_df.empty:
            return f"No price information or configurations found for sofa ID '{sofa_id_str}' based on the criteria."
        results_list = []
        for _, row in price_df.iterrows():
            # Construct display string for the configuration modules
            config_parts = [f"Config_{i + 1}: {row[col]}" for i, col in enumerate(self.config_cols) if
                            col in row and pd.notna(row[col])]
            config_display_str = "; ".join(config_parts) if config_parts else "Base Configuration"
            price_str = f"Sell-out Price: {row['Sell_out']:,.0f} EUR" if 'Sell_out' in row and pd.notna(row['Sell_out']) else "Price N/A"
            sofa_code_display = row.get('Codice', sofa_id_str) # Use Codice from row or passed sofa_id_str

            # Attempt to fetch sofa name if not already in the price_df row
            sofa_name_val = row.get('Nome')
            if pd.isna(sofa_name_val) and hasattr(self, '_sofa_info'): # Check if _sofa_info is loaded
                sofa_info_match = self._sofa_info[self._sofa_info['Codice'] == str(sofa_code_display)]
                if not sofa_info_match.empty: sofa_name_val = sofa_info_match['Nome'].iloc[0]
            sofa_name_display = f" (Name: {sofa_name_val})" if pd.notna(sofa_name_val) else ""

            results_list.append(
                f"Sofa ID {sofa_code_display}{sofa_name_display} with {config_display_str} -> {price_str}")
        if not results_list: return f"No price information found for sofa ID '{sofa_id_str}' matching the criteria."
        return "\n".join(results_list)

    def get_sofa_predefined_configs(self, sofa_id_str: str) -> str:
        """
        Retrieves predefined configurations, including total seats and price, for a given sofa ID
        from the dedicated predefined configurations CSV file.
        Args:
            sofa_id_str (str): The ID of the sofa.
        Returns:
            str: Formatted string of predefined configurations or an error message.
        """
        if not hasattr(self, '_predefined_configs_info'):
            return "Error: Predefined configurations data (price_seats_info) not loaded."

        sofa_id_str = str(sofa_id_str).strip()
        # Filter the predefined configurations DataFrame for the given sofa ID
        config_data = self._predefined_configs_info[self._predefined_configs_info['ID'] == sofa_id_str].copy()

        if config_data.empty:
            return f"No predefined configurations found for sofa ID '{sofa_id_str}' in the showcase file."

        sofa_name = config_data['Nome'].iloc[0] # Assumes all rows for an ID will have the same name
        results = [f"Predefined configurations for sofa '{sofa_name}' (ID: {sofa_id_str}):"]

        version_cols = [f'Ver{i}' for i in range(1, 6)] # Columns like Ver1, Ver2...

        for _, row in config_data.iterrows():
            # Collect module codes for the current configuration
            config_modules = [str(row[col]) for col in version_cols if pd.notna(row[col]) and row[col]!=''] # Ensure modules are strings and handle potential floats
            config_str = "Modules: " + ", ".join(config_modules)

            seats = row.get('Total_Seats', 'N/A')
            price = row.get('Total_Price_EUR', 'N/A')

            # Format price and seats for display
            price_str = f"{price:,.0f} EUR" if pd.notna(price) else "N/A"
            seats_str = f"{seats:.1f}" if pd.notna(seats) else "N/A"

            results.append(f"  - {config_str} -> Total Seats: {seats_str}, Total Price: {price_str}")

        return "\n".join(results)

    def _format_sofa_search_results(self, df_results: pd.DataFrame, N: int = 5) -> str:
        """
        Helper function to format sofa search results into a readable string.
        Args:
            df_results (pd.DataFrame): DataFrame containing the sofas that matched the query.
            N (int, optional): The maximum number of random results to display. Defaults to 5.
        Returns:
            str: A formatted string of search results.
        """
        if df_results.empty: return "No sofas found matching the criteria."
        results = []
        sample_n = min(N, len(df_results)) # Determine number of items to display
        # Sample N items if more than N found, otherwise display all
        df_to_display = df_results.sample(n=sample_n, random_state=42) if len(
            df_results) > sample_n else df_results.sample(frac=1, random_state=42).head(sample_n)

        for _, row in df_to_display.iterrows():
            # Construct a descriptive string for each sofa
            name_str = f" (Name: {row['Nome']})" if pd.notna(row.get('Nome')) else ""
            designer_str = f" - Designer: {row['Designer']}" if pd.notna(row.get('Designer')) and str(
                row.get('Designer')).lower() != 'non specificato' else ""
            material_str = f" - Material: {row['Materiale_Rivestimento']}" if pd.notna(
                row.get('Materiale_Rivestimento')) else ""
            modular_str = " - Modular" if pd.notna(row.get('Modulare')) and str(
                row.get('Modulare')).lower() == 'sì' else ""
            bed_func_str = " - Bed Function" if pd.notna(row.get('Funzione_Letto')) and str(
                row.get('Funzione_Letto')).lower() == 'sì' else ""
            results.append(f"ID: {row['Codice']}{name_str}{designer_str}{material_str}{modular_str}{bed_func_str}")
        return "\n".join(results) if results else "No results to format."

    def _parse_conditions(self, condition_str: str, df_to_filter: pd.DataFrame) -> Tuple[pd.Series, str]:
        """
        Parses a string of combined conditions (ANDed) and applies them to a DataFrame.
        Args:
            condition_str (str): The condition string (e.g., "feature=value && feature2 CONTAINS value2").
            df_to_filter (pd.DataFrame): The DataFrame to filter.
        Returns:
            Tuple[pd.Series, str]: A boolean Series for filtering and an error message string (empty if no error).
        """
        conditions = condition_str.split('&&')
        combined_filter = pd.Series([True] * len(df_to_filter), index=df_to_filter.index) # Start with all True

        known_numeric_cols = ['sell_out', 'prezzo', 'piedi_altezza_cm', 'anno_nascita', 'total_seats', 'seats'] # Columns known to be numeric

        for cond in conditions:
            cond = cond.strip()
            match = re.match(r"^\s*([\w\s.-]+)\s*([!=<>]=?|CONTAINS)\s*(.+)\s*", cond, re.IGNORECASE) # Regex for "col op val"
            if not match:
                return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Invalid condition format: '{cond}'"

            column_name, operator, value_str = match.groups()
            column_name, operator, value_str = column_name.strip(), operator.strip().upper(), value_str.strip().strip("'\"")

            actual_col_name = next((col for col in df_to_filter.columns if col.lower() == column_name.lower()), None) # Case-insensitive column match
            if not actual_col_name:
                return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Column '{column_name}' not found."

            target_column = df_to_filter[actual_col_name]
            try:
                current_filter = pd.Series([False] * len(df_to_filter), index=df_to_filter.index)
                # Determine if comparison should be numeric
                is_numeric_comparison = actual_col_name.lower() in known_numeric_cols or pd.api.types.is_numeric_dtype(target_column.dtype)

                if is_numeric_comparison:
                    try:
                        value = float(value_str)
                        target_numeric = pd.to_numeric(target_column, errors='coerce')
                    except ValueError: # If value for numeric comparison isn't a number
                        return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Cannot convert '{value_str}' to number for '{actual_col_name}'."

                    # Apply numeric operators
                    if operator == '=' or operator == '==': current_filter = (target_numeric == value)
                    elif operator == '!=': current_filter = (target_numeric != value)
                    elif operator == '<': current_filter = (target_numeric < value)
                    elif operator == '>': current_filter = (target_numeric > value)
                    elif operator == '<=': current_filter = (target_numeric <= value)
                    elif operator == '>=': current_filter = (target_numeric >= value)
                    else: return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Operator '{operator}' not valid for numeric feature '{actual_col_name}'."
                    current_filter = current_filter.fillna(False) # NaN comparisons result in False
                else:  # String operations
                    value = str(value_str).lower()
                    if operator == '=' or operator == '==': current_filter = (target_column.astype(str).str.lower() == value)
                    elif operator == '!=': current_filter = (target_column.astype(str).str.lower() != value)
                    elif operator == 'CONTAINS': current_filter = target_column.astype(str).str.lower().str.contains(value, na=False)
                    else: return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Operator '{operator}' not valid for text feature '{actual_col_name}'. Use '=', '!=', or 'CONTAINS'."
                combined_filter &= current_filter # AND with previous conditions
            except Exception as e:
                return pd.Series([False] * len(df_to_filter), index=df_to_filter.index), f"Error applying condition '{cond}': {e}"
        return combined_filter, ""

    def sofa_feature_query(self, condition_str: str) -> str:
        """
        Queries sofas based on feature conditions from the main sofa catalog.
        Args:
            condition_str (str): The feature-based condition string (e.g., "Materiale_Rivestimento=Pelle").
        Returns:
            str: Formatted search results or an error message.
        """
        if not hasattr(self, '_sofa_info') or self._sofa_info.empty:
            return "Error: Sofa catalog (sofa_info) is not loaded or empty."
        df_to_filter = self._sofa_info.copy()
        # If no condition, return random sofas
        if not condition_str or not condition_str.strip() or condition_str.strip() in ["''", '""']:
            sample_size = min(5, len(df_to_filter))
            if sample_size == 0: return "No sofas in catalog to sample for empty query."
            random_sofas = df_to_filter.sample(n=sample_size, random_state=42)
            return "No specific search criteria. Here are 5 random sofas:\n" + self._format_sofa_search_results(random_sofas, N=sample_size)

        final_filter, error_msg = self._parse_conditions(condition_str, df_to_filter)
        if error_msg: return f"Error parsing feature conditions: {error_msg}"
        matching_sofas = df_to_filter[final_filter]
        if matching_sofas.empty: return f"No sofas found matching: {condition_str}"

        num_found = len(matching_sofas)
        display_count = min(num_found, 5)
        # Sample for display if more results than N
        matching_sofas_display = matching_sofas.sample(n=display_count, random_state=42) if num_found > display_count else matching_sofas.sample(frac=1, random_state=42)

        formatted_results = self._format_sofa_search_results(matching_sofas_display, N=display_count)
        output_intro = f"Found {num_found} sofa(s) matching '{condition_str}'. "
        output_intro += f"Showing {display_count} random results:\n" if num_found > display_count else "Details:\n" if num_found > 0 else ""
        return output_intro + formatted_results

    def sofa_price_query(self, price_condition_str: str) -> str:
        """
        Queries sofa configurations based on price conditions.
        Args:
            price_condition_str (str): The price-based condition string (e.g., "Sell_out < 3000").
        Returns:
            str: Formatted search results or an error message.
        """
        if not hasattr(self, '_price_info') or self._price_info.empty:
            return "Error: Sofa prices catalog (price_info) not loaded or empty."
        df_to_filter = self._price_info.copy()
        df_to_filter['Sell_out'] = pd.to_numeric(df_to_filter['Sell_out'], errors='coerce') # Ensure price is numeric
        df_to_filter.dropna(subset=['Sell_out'], inplace=True) # Remove rows without valid price

        # If no condition, return random priced configurations
        if not price_condition_str or not price_condition_str.strip() or price_condition_str.strip() in ["''", '""']:
            sample_size = min(5, len(df_to_filter))
            if sample_size == 0: return "No sofa configurations with price to sample for empty query."
            random_prices = df_to_filter.sample(n=sample_size, random_state=42)
            # Merge with sofa names for better display
            random_prices_with_names = pd.merge(random_prices, self._sofa_info[['Codice', 'Nome']].drop_duplicates(subset=['Codice']), on='Codice', how='left')
            return "No specific price criteria. Here are 5 random sofa configurations with prices:\n" + self._format_price_results(random_prices_with_names)

        final_filter, error_msg = self._parse_conditions(price_condition_str, df_to_filter)
        if error_msg: return f"Error parsing price conditions: {error_msg}"
        matching_prices = df_to_filter[final_filter]
        if matching_prices.empty: return f"No sofa configurations found matching price conditions: {price_condition_str}"

        num_found = len(matching_prices)
        display_count = min(num_found, 5)
        matching_prices_display = matching_prices.sample(n=display_count, random_state=42) if num_found > display_count else matching_prices.sample(frac=1, random_state=42)

        # Merge with sofa names for display
        matching_prices_with_names = pd.merge(matching_prices_display, self._sofa_info[['Codice', 'Nome']].drop_duplicates(subset=['Codice']), on='Codice', how='left')
        formatted_results = self._format_price_results(matching_prices_with_names)
        output_intro = f"Found {num_found} sofa configuration(s) matching '{price_condition_str}'. "
        output_intro += f"Showing {display_count} random results:\n" if num_found > display_count else "Details:\n" if num_found > 0 else ""
        return output_intro + formatted_results

    def sofa_seats_query(self, seats_condition_str: str) -> str:
        """
        Filters sofas based on the number of seats in their predefined configurations
        using the '_predefined_configs_info' DataFrame.
        Args:
            seats_condition_str (str): The seat-based condition string (e.g., "Total_Seats >= 3").
        Returns:
            str: Formatted search results of sofa models or an error message.
        """
        if not hasattr(self, '_predefined_configs_info') or self._predefined_configs_info.empty:
            return "Error: Predefined configurations data (price_seats_info) not loaded, cannot search by seats."

        df_to_filter = self._predefined_configs_info.copy() # Use the configurations data which has Total_Seats

        # Parse the seat condition against the predefined configurations data
        final_filter, error_msg = self._parse_conditions(seats_condition_str, df_to_filter)
        if error_msg:
            return f"Error parsing seats condition: {error_msg}"

        matching_configs = df_to_filter[final_filter]
        if matching_configs.empty:
            return f"No sofa configurations found matching seat condition: {seats_condition_str}"

        matching_sofa_ids = matching_configs['ID'].unique().tolist() # Get unique sofa IDs from matching configs

        # Retrieve full details for these sofa IDs from the main sofa catalog
        matching_sofas_df = self._sofa_info[self._sofa_info['Codice'].isin(matching_sofa_ids)]

        num_found = len(matching_sofas_df)
        output_intro = f"Found {num_found} sofa model(s) with configurations matching '{seats_condition_str}'. "
        return output_intro + self._format_sofa_search_results(matching_sofas_df)

    def sofa_combo_query(self, complex_condition_str: str) -> str:
        """
        Queries sofas based on a combination of feature, price, and seat conditions.
        Args:
            complex_condition_str (str): The combined condition string.
        Returns:
            str: Formatted search results or an error message.
        """
        if not (hasattr(self, '_sofa_info') and hasattr(self, '_price_info') and hasattr(self, '_predefined_configs_info')):
            return "Error: Required data for combo query (catalog, prices, or predefined configs) is not loaded."

        all_conditions = complex_condition_str.split('&&')
        feature_conditions, price_conditions, seat_conditions = [], [], []

        # Categorize conditions
        for cond in all_conditions:
            cond_strip = cond.strip()
            # Check for price-related keywords or typical price operators
            if re.search(r'\b(?:sell_out|prezzo)\b', cond_strip, re.IGNORECASE): # Price condition
                price_conditions.append(cond_strip)
            # Check for seat-related keywords
            elif re.search(r'\b(?:total_seats|posti|sedute)\b', cond_strip, re.IGNORECASE): # Seat condition
                seat_conditions.append(cond_strip)
            else: # Otherwise, assume feature condition
                feature_conditions.append(cond_strip)

        # 1. Filter by features first (from main sofa catalog)
        df_feat_filtered = self._sofa_info.copy()
        if feature_conditions:
            feat_cond_str = " && ".join(feature_conditions)
            feat_filter, err_msg = self._parse_conditions(feat_cond_str, df_feat_filtered)
            if err_msg: return f"Error parsing feature conditions: {err_msg}"
            df_feat_filtered = df_feat_filtered[feat_filter]
        if df_feat_filtered.empty: return f"No sofas found for features: {' && '.join(feature_conditions) or 'N/A'}"

        # 2. Filter by seats (from predefined configurations, then map back to sofa IDs)
        df_seats_filtered = df_feat_filtered.copy() # Start with feature-filtered sofas
        if seat_conditions:
            seat_cond_str = " && ".join(seat_conditions)
            # Parse seat conditions against the _predefined_configs_info DataFrame
            seat_filter, err_msg = self._parse_conditions(seat_cond_str, self._predefined_configs_info)
            if err_msg: return f"Error parsing seat conditions: {err_msg}"
            # Get IDs of sofas that have at least one configuration matching the seat criteria
            sofa_ids_matching_seats = self._predefined_configs_info[seat_filter]['ID'].unique()
            # Further filter the feature-filtered sofas by these IDs
            df_seats_filtered = df_seats_filtered[df_seats_filtered['Codice'].isin(sofa_ids_matching_seats)]
        if df_seats_filtered.empty: return f"Sofas matched features, but none met the seat criteria: {' && '.join(seat_conditions)}"

        # 3. Filter by price (from price_info, then map back to sofa IDs)
        df_price_filtered = df_seats_filtered.copy() # Start with feature-and-seat-filtered sofas
        if price_conditions:
            matching_ids_for_price_check = df_price_filtered['Codice'].unique()
            prices_df_for_cond_check = self._price_info[self._price_info['Codice'].isin(matching_ids_for_price_check)].copy()
            if prices_df_for_cond_check.empty:
                return f"Sofas matched features/seats, but no price entries found for them to check price conditions: {' && '.join(price_conditions)}"

            price_cond_str = " && ".join(price_conditions)
            # Parse price conditions against the relevant subset of _price_info
            price_filter, err_msg = self._parse_conditions(price_cond_str, prices_df_for_cond_check)
            if err_msg: return f"Error parsing price conditions: {err_msg}"
            # Get IDs of sofas that have at least one configuration matching the price criteria
            sofa_ids_matching_price = prices_df_for_cond_check[price_filter]['Codice'].unique()
            # Further filter by these IDs
            df_price_filtered = df_price_filtered[df_price_filtered['Codice'].isin(sofa_ids_matching_price)]

        if df_price_filtered.empty: return f"No sofas found for the combined criteria: {complex_condition_str}"

        return f"Found {len(df_price_filtered)} sofa(s) matching all criteria. Showing up to 5:\n" + self._format_sofa_search_results(
            df_price_filtered)