"""
Plugin System Tab for the Data Analysis Toolkit

Provides a Streamlit UI for loading, managing, and executing plugins.
"""

import os
import sys

import streamlit as st
import pandas as pd

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from plugin_system import (
    PluginManager, get_example_plugins, get_plugin_template
)

PLOTLY_TEMPLATE = "plotly_white"


def _get_plugin_manager():
    """Get or create the plugin manager in session state."""
    if 'plugin_manager' not in st.session_state:
        st.session_state.plugin_manager = PluginManager()
    return st.session_state.plugin_manager


def render_plugin_tab():
    """Render the Plugin System tab."""
    st.header("🔌 Plugin System")

    with st.expander("ℹ️ About Plugins", expanded=False):
        st.markdown("""
        ### 🔌 Extend the Toolkit with Custom Plugins

        Plugins allow you to add custom analysis, preprocessing, or visualization
        functions without modifying the core toolkit.

        **How it works:**
        1. **Load** a plugin from a `.py` file or paste code directly
        2. **Configure** parameters via auto-generated forms
        3. **Execute** on your loaded data
        4. **Save** plugins for reuse across sessions

        ---

        ### 📋 Plugin Structure

        Every plugin file needs three things:

        ```python
        # 1. Metadata dict
        PLUGIN_INFO = {
            "name": "My Custom Analysis",
            "description": "What the plugin does",
            "category": "analysis",     # analysis | preprocessing | visualization
            "version": "1.0",
            "author": "Your Name"
        }

        # 2. Parameters dict (auto-generates UI controls)
        PLUGIN_PARAMETERS = {
            "threshold": {"type": "float", "default": 0.05,
                          "min": 0.0, "max": 1.0,
                          "description": "Significance threshold"},
            "method":    {"type": "select",
                          "options": ["pearson", "spearman"],
                          "default": "pearson",
                          "description": "Correlation method"}
        }

        # 3. Processing function
        def process(data, columns, target=None, **params):
            # data: pandas DataFrame
            # columns: list of selected feature columns
            # target: optional target column name
            # params: values from PLUGIN_PARAMETERS
            result = ...
            return {"summary": "...", "data": result}
        ```

        ---

        ### 📦 Bundled Example Plugins

        | Plugin | Category | Description |
        |--------|----------|-------------|
        | **Enhanced Scatter** | Visualization | Scatter plot with regression, marginals |
        | **Lag Features** | Preprocessing | Generate lagged copies of time series columns |
        | **Outlier Removal** | Preprocessing | IQR / Z-score based outlier removal |

        Load these from the **Example templates** option.

        ---

        ### 💡 Tips

        - Parameter types supported: `float`, `int`, `bool`, `str`, `select`
        - Return a dict with a `"data"` key (DataFrame) to make results downloadable
        - Plugins run in an isolated namespace — they cannot modify the core toolkit
        - Check the `example_plugins/` folder for ready-to-use templates
        """)

    pm = _get_plugin_manager()

    # --- Three-column layout: Load | Manage | Execute ---
    load_col, exec_col = st.columns([1, 1])

    # =====================================================================
    # LEFT: Load Plugins
    # =====================================================================
    with load_col:
        st.subheader("📥 Load Plugin")

        load_method = st.radio(
            "Load method:",
            ["📁 From file", "📋 Paste code", "📦 Example templates"],
            horizontal=True,
            key="plugin_load_method"
        )

        if load_method == "📁 From file":
            uploaded = st.file_uploader(
                "Upload a `.py` plugin file",
                type=["py"],
                key="plugin_file_upload"
            )
            if uploaded is not None:
                code = uploaded.getvalue().decode('utf-8')
                if st.button("🔄 Load Plugin", key="load_from_upload"):
                    plugin, msg = pm.load_from_code(code, source_path=uploaded.name)
                    if plugin:
                        st.success(msg)
                    else:
                        st.error(msg)

            # Also allow loading from filesystem path
            file_path = st.text_input(
                "Or enter a file path:",
                placeholder="/path/to/my_plugin.py",
                key="plugin_file_path"
            )
            if file_path and st.button("📂 Load from Path", key="load_from_path"):
                plugin, msg = pm.load_from_file(file_path)
                if plugin:
                    st.success(msg)
                else:
                    st.error(msg)

        elif load_method == "📋 Paste code":
            template = get_plugin_template()
            code = st.text_area(
                "Plugin source code:",
                value=template,
                height=400,
                key="plugin_code_editor"
            )
            if st.button("🔄 Load Plugin from Code", key="load_from_code"):
                plugin, msg = pm.load_from_code(code)
                if plugin:
                    st.success(msg)
                else:
                    st.error(msg)

        else:  # Example templates
            examples = get_example_plugins()
            selected = st.selectbox(
                "Choose an example:",
                list(examples.keys()),
                key="plugin_example_select"
            )
            if selected:
                st.code(examples[selected], language="python")
                if st.button(f"🔄 Load '{selected}'", key="load_example"):
                    plugin, msg = pm.load_from_code(examples[selected])
                    if plugin:
                        st.success(msg)
                    else:
                        st.error(msg)

    # =====================================================================
    # RIGHT: Execute Plugin
    # =====================================================================
    with exec_col:
        st.subheader("▶️ Execute Plugin")

        plugins = pm.get_all_plugins()
        if not plugins:
            st.info("No plugins loaded. Load one from the left panel.")
        else:
            plugin_names = {p.id: f"{p.info.name} (v{p.info.version})" for p in plugins}
            selected_id = st.selectbox(
                "Select plugin:",
                list(plugin_names.keys()),
                format_func=lambda x: plugin_names[x],
                key="plugin_exec_select"
            )

            plugin = pm.get_plugin(selected_id)
            if plugin:
                # Show plugin info
                st.caption(f"📝 {plugin.info.description}")
                st.caption(f"👤 {plugin.info.author} | 📁 {plugin.info.category}")

                # Auto-generate parameter form
                param_values = {}
                if plugin.parameters:
                    st.markdown("**Parameters:**")
                    for name, param in plugin.parameters.items():
                        if param.type == 'float':
                            param_values[name] = st.number_input(
                                f"{param.description or name}",
                                value=float(param.default or 0),
                                min_value=float(param.min) if param.min is not None else None,
                                max_value=float(param.max) if param.max is not None else None,
                                key=f"plugin_param_{selected_id}_{name}"
                            )
                        elif param.type == 'int':
                            param_values[name] = st.number_input(
                                f"{param.description or name}",
                                value=int(param.default or 0),
                                min_value=int(param.min) if param.min is not None else None,
                                max_value=int(param.max) if param.max is not None else None,
                                step=1,
                                key=f"plugin_param_{selected_id}_{name}"
                            )
                        elif param.type == 'bool':
                            param_values[name] = st.checkbox(
                                f"{param.description or name}",
                                value=bool(param.default),
                                key=f"plugin_param_{selected_id}_{name}"
                            )
                        elif param.type == 'str' and param.choices:
                            param_values[name] = st.selectbox(
                                f"{param.description or name}",
                                param.choices,
                                index=param.choices.index(param.default) if param.default in param.choices else 0,
                                key=f"plugin_param_{selected_id}_{name}"
                            )
                        else:
                            param_values[name] = st.text_input(
                                f"{param.description or name}",
                                value=str(param.default or ''),
                                key=f"plugin_param_{selected_id}_{name}"
                            )

                # Column selection
                df = st.session_state.get('df')
                if df is not None:
                    numeric_cols = df.select_dtypes(include='number').columns.tolist()
                    all_cols = df.columns.tolist()

                    columns = st.multiselect(
                        "Feature columns:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                        key="plugin_columns"
                    )
                    target = st.selectbox(
                        "Target column (optional):",
                        [None] + all_cols,
                        key="plugin_target"
                    )

                    if st.button("▶️ Run Plugin", type="primary", key="run_plugin"):
                        with st.spinner(f"Running {plugin.info.name}..."):
                            result, msg = plugin.execute(
                                df, columns=columns, target=target, **param_values
                            )

                        st.markdown(f"**Result:** {msg}")

                        if result is not None:
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result.head(100))
                                if st.button("💾 Replace data with result", key="plugin_replace"):
                                    st.session_state.df = result
                                    st.success("Data replaced!")
                                    st.rerun()
                            elif isinstance(result, dict):
                                st.json(result)
                            else:
                                st.write(result)
                else:
                    st.warning("📁 Please load data first in the Data tab.")

    # =====================================================================
    # BOTTOM: Manage Loaded Plugins
    # =====================================================================
    st.markdown("---")
    st.subheader("📋 Loaded Plugins")

    plugins = pm.get_all_plugins()
    if not plugins:
        st.info("No plugins loaded yet.")
    else:
        for plugin in plugins:
            with st.expander(
                f"{'✅' if plugin.enabled else '❌'} {plugin.info.name} "
                f"v{plugin.info.version} ({plugin.info.category})"
            ):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{plugin.info.description}**")
                    st.caption(f"Author: {plugin.info.author} | ID: {plugin.id}")
                    if plugin.parameters:
                        st.caption(f"Parameters: {', '.join(plugin.parameters.keys())}")
                with col2:
                    if st.button("💾 Save", key=f"save_{plugin.id}"):
                        ok, msg = pm.save_plugin(plugin.id)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                with col3:
                    if st.button("🗑️ Unload", key=f"unload_{plugin.id}"):
                        pm.unload_plugin(plugin.id)
                        st.rerun()

                # Show source code
                with st.expander("📄 Source Code"):
                    st.code(plugin.source_code, language="python")
