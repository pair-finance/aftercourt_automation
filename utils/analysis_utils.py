

def get_token_frequency(data, tokenized_column='tokenized', normalize=True, sort=True):
    """
    Calculate token frequency across documents.
    
    Args:
        data: DataFrame containing the tokenized text column
        tokenized_column: Name of the column containing tokenized text (default: 'tokenized')
        normalize: If True, normalize frequencies by number of documents (default: True)
        sort: If True, sort by frequency in descending order (default: True)
    
    Returns:
        Dictionary mapping tokens to their frequencies
    """
    vocabulary = {}
    total_documents = len(data)
    
    for tokens in data[tokenized_column]:
        for token in tokens:
            if token in vocabulary:
                vocabulary[token] += 1
            else:
                vocabulary[token] = 1
    
    if normalize:
        for token in vocabulary:
            vocabulary[token] = round(vocabulary[token] / total_documents, 2)
    
    if sort:
        vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True))
    
    return vocabulary


def plot_token_counts_by_document_type(data, text_column='text_w_tags', document_type_column='document_type', 
                                       special_tokens=None, highlight_doc_types=None, figsize=(12, 6)):
    """
    Plot a scatter plot showing the counts of each special token per document type.
    
    Args:
        data: DataFrame containing the text and document type columns
        text_column: Name of the column containing text with tags (default: 'text_w_tags')
        document_type_column: Name of the column containing document types (default: 'document_type')
        special_tokens: Set of special tokens to count (default: standard set of tokens)
        highlight_doc_types: List of document types to highlight with 'X' marker (default: None)
        figsize: Figure size as tuple (width, height) (default: (12, 6))
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    if special_tokens is None:
        special_tokens = {"<LEGAL_ARTICLE>", "<DR_REFERENCE>", "<CUR>", "<URL>", "<EMAIL>", "<PHONE>", "<NUMBER>"}
    
    if highlight_doc_types is None:
        highlight_doc_types = []
    
    # Count occurrences of each token per document type
    token_counts = []
    for token in special_tokens:
        for doc_type in data[document_type_column].unique():
            count = data[data[document_type_column] == doc_type][text_column].str.count(token).sum()
            token_counts.append({
                'token': token,
                'document_type': doc_type,
                'count': np.log(count + 1)  # use log scale for better visualization
            })
    
    token_df = pd.DataFrame(token_counts)
    
    # Create scatter plot
    plt.figure(figsize=figsize)
    for doc_type in token_df['document_type'].unique():
        subset = token_df[token_df['document_type'] == doc_type]
        if doc_type in highlight_doc_types:
            plt.scatter(subset['token'], subset['count'], label=doc_type, alpha=0.7, s=100, marker='X')
        else:
            plt.scatter(subset['token'], subset['count'], label=doc_type, alpha=0.7, s=100)
    
    plt.xlabel('Token Name')
    plt.ylabel('Log Count')
    plt.title('Token Counts by Document Type')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def check_specific_token(data, token, tokenized_column='tokenized', filter_column=None, filter_value=None):
    """
    Check and filter data for rows containing a specific token.
    
    Args:
        data: DataFrame containing the tokenized text column
        token: The token to search for
        tokenized_column: Name of the column containing tokenized text (default: 'tokenized')
        filter_column: Optional column name to apply additional filter (default: None)
        filter_value: Value for the filter column (default: None)
    
    Returns:
        Filtered DataFrame containing rows with the specified token
    
    Example:
        # Find rows with token "mo" and is_pfub == 1
        data_with_mo = check_specific_token(data, 'mo', filter_column='is_pfub', filter_value=1)
    """
    # Filter for rows containing the token
    data_with_token = data[data[tokenized_column].apply(lambda x: token in x)]
    
    # Apply additional filter if specified
    if filter_column is not None and filter_value is not None:
        data_with_token = data_with_token[data_with_token[filter_column] == filter_value]
    
    return data_with_token


def highlight_token_in_text(text, token, color_code=91):
    """
    Highlight a specific token in text with color.
    
    Args:
        text: The text to highlight
        token: The token to highlight
        color_code: ANSI color code (default: 91 for red)
            Common codes: 91=red, 92=green, 93=yellow, 94=blue, 95=magenta, 96=cyan
    
    Returns:
        Text with highlighted token
    
    Example:
        print(highlight_token_in_text("mo. fr.: 08:00 12.00 uhr", "mo"))
    """
    return text.replace(token, f"\033[{color_code}m{token}\033[0m")


def inspect_token_in_data(data, token, text_column='cleaned_text', tokenized_column='tokenized', 
                          filter_column=None, filter_value=None, row_index=0, color_code=91):
    """
    Inspect and print text containing a specific token with highlighting.
    
    Args:
        data: DataFrame containing the data
        token: The token to search for and highlight
        text_column: Name of the column containing text to display (default: 'cleaned_text')
        tokenized_column: Name of the column containing tokenized text (default: 'tokenized')
        filter_column: Optional column name to apply additional filter (default: None)
        filter_value: Value for the filter column (default: None)
        row_index: Index of the row to display (default: 0)
        color_code: ANSI color code for highlighting (default: 91 for red)
    
    Returns:
        Filtered DataFrame and prints the specified row with highlighted token
    
    Example:
        # Check token "mo" in data where is_pfub == 1, show 15th row
        filtered_data = inspect_token_in_data(data, 'mo', filter_column='is_pfub', 
                                              filter_value=1, row_index=15)
    """
    # Get filtered data
    filtered_data = check_specific_token(data, token, tokenized_column, filter_column, filter_value)
    
    if len(filtered_data) == 0:
        print(f"No data found with token '{token}'")
        return filtered_data
    
    if row_index >= len(filtered_data):
        print(f"Row index {row_index} is out of bounds. Dataset has {len(filtered_data)} rows.")
        return filtered_data
    
    # Get and print text with highlighted token
    text = filtered_data.iloc[row_index][text_column]
    highlighted_text = highlight_token_in_text(text, token, color_code)
    print(highlighted_text)
    
    return filtered_data