import pandas as pd
from IPython.display import display


def Table(title, headers, data):
    """
    Display a table in a Jupyter notebook using pandas DataFrame.

    Args:
        title: Name of table
        headers (list): A list of strings containing the column headers.
        data (list): A list of lists containing the data by row to display in the table.

    Example:
        headers = ["Token", "Frequency"]
        data = [
            ["apple", 100],
            ["banana", 50],
            ["cherry", 25],
        ]
        Table("Fruits", headers, data)
    """

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Define custom styles
    styles = [
        {
            "selector": "th",
            "props": [
                ("background-color", "#dddddd"),
                ("color", "black"),
                ("text-align", "center"),
                ("padding", "10px"),
                ("font-size", "16px"),
                ("border-bottom", "2px solid #ccc"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("padding", "8px"),
                ("font-size", "14px"),
                ("border-bottom", "1px solid #ccc"),
                ("color", "black"),
            ],
        },
        {"selector": "tr:nth-child(odd)", "props": [("background-color", "#f9f9f9")]},
        {"selector": "tr:nth-child(even)", "props": [("background-color", "white")]},
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("width", "50%"),
                ("margin", "25px 0"),
                ("font-family", "Arial, sans-serif"),
                ("border-radius", "8px"),
                ("overflow", "hidden"),
            ],
        },
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("font-size", "18px"),
                ("font-weight", "bold"),
                ("margin-bottom", "10px"),
            ],
        },
    ]

    # Additional styles for hover effect and column width
    hover_styles = [
        {"selector": "tbody tr:hover", "props": [("background-color", "#e0e0e0")]}
    ]

    # Apply the styles to the DataFrame and add a title
    styled_df = df.style.set_table_styles(styles).set_caption(title)
    styled_df = styled_df.hide(axis="index")
    styled_df = styled_df.set_table_styles(hover_styles, overwrite=False)
    display(styled_df)
