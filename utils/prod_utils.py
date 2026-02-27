import json
import os
from typing import Any, Dict, List

import boto3
import pandas as pd


def get_texts_from_textract_outputs(textract_outputs: List[Dict[str, Any]]) -> List[str]:
    raw_text_outputs = []
    for output in textract_outputs:
        if output is None:
            raw_text_outputs.append("")
        else:
            raw_text = []
            for cur_doc in output:
                if cur_doc["BlockType"] == "LINE":
                    raw_text.append(cur_doc["Text"])    
               
            raw_text_outputs.append("\n".join(raw_text))

    return raw_text_outputs


def show_predictions(data: pd.DataFrame) -> None:
    """Display predictions in a formatted, readable way with colors."""
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}📊 PREDICTIONS SUMMARY ({len(data)} predictions){RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")
    
    for idx, (_, row) in enumerate(data.iterrows(), 1):
        print(f"{GRAY}┌─ {BOLD}Prediction #{idx}{RESET} {GRAY}{'─' * (45 - len(str(idx)))}{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}🏷️  Type:{RESET}       {GREEN}{row['type']}{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}📋 Subtype:{RESET}    {GREEN}{row['subtype']}{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}✨ Value:{RESET}      {MAGENTA}{row['value']}{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}🤖 Model:{RESET}      {BLUE}{row['model_name']}{RESET}")
        print(f"{GRAY}└{'─' * 58}{RESET}\n")
    
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")


def get_data_by_egvp_id(egvp_id: str, analytics_db, s3, pdf_download: bool = False, pdf_download_dir: str = None, verbose: bool = True) -> pd.DataFrame:
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    query = f"""
        SELECT 
            llm_tickets.ticket_uuid,
            llm_tickets.source_type,
            llm_tickets.egvp_id,
            llm_tickets.status,
            llm_tickets.origin,
            llm_attachments.attachment_id,
            llm_attachments.s3_key as document_s3_key,
            llm_attachments.s3_bucket as document_s3_bucket,
            llm_attachments.file_name,
            llm_attachments_predictions.model_name,
            llm_attachments_predictions.type,
            llm_attachments_predictions.subtype,
            llm_attachments_predictions.value,
            textract_jobs.job_id AS textract_job_id,
            textract_jobs.status AS textract_status,
            textract_jobs.s3_link AS textract_s3_link
        FROM llm_tickets
        JOIN llm_attachments ON llm_tickets.ticket_uuid = llm_attachments.ticket_uuid
        JOIN llm_attachments_predictions ON llm_attachments.attachment_id = llm_attachments_predictions.attachment_id
        JOIN textract_jobs ON llm_attachments.attachment_id = textract_jobs.attachment_id
        WHERE llm_tickets.egvp_id = '{egvp_id}'
        AND llm_tickets.source_type = 'egvp'
        ORDER BY llm_tickets.created_at DESC
    """
    data = analytics_db.sql_to_df(query)
    cols = [
        'ticket_uuid', 'source_type', 'egvp_id', 'status', 'origin',
        'attachment_id', 'document_s3_key', 'document_s3_bucket', 'file_name', 'model_name', 'type', 'subtype', 'value',
        'textract_job_id', 'textract_status', 'textract_s3_link'
    ]
    
    # Header
    if verbose:
        print(f"\n{CYAN}{BOLD}{'='*100}{RESET}")
        print(f"{CYAN}{BOLD}🎫 TICKET SUMMARY{RESET}")
        print(f"{CYAN}{BOLD}{'='*100}{RESET}")
        print(f"{YELLOW}Ticket UUID:{RESET}     {GREEN}{data['ticket_uuid'].iloc[0]}{RESET}")
        print(f"{YELLOW}EGVP ID:{RESET}         {GREEN}{data['egvp_id'].iloc[0]}{RESET}")
        print(f"{YELLOW}Status:{RESET}          {BLUE}{data['status'].iloc[0]}{RESET}")
        print(f"{YELLOW}Origin:{RESET}          {BLUE}{data['origin'].iloc[0]}{RESET}")
        print(f"{YELLOW}Total Attachments:{RESET} {MAGENTA}{len(data['attachment_id'].unique())}{RESET}")
        print(f"{CYAN}{BOLD}{'='*100}{RESET}\n")
    
    unique_attachments = data['attachment_id'].unique()
    document_s3_buckets= data['document_s3_bucket']
    document_s3_keys = data['document_s3_key']
    textract_text_s3_links = data['textract_s3_link']
    
    for idx, attch in enumerate(unique_attachments, 1):
        if verbose:
            print(f"{GRAY}┌{'─' * 98}{RESET}")
            print(f"{GRAY}│{RESET} {CYAN}{BOLD}📎 ATTACHMENT {idx}/{len(unique_attachments)}{RESET}")
            print(f"{GRAY}├{'─' * 98}{RESET}")
        
        document_s3_bucket = data[data['attachment_id'] == attch]['document_s3_bucket'].values[0]
        document_s3_key = data[data['attachment_id'] == attch]['document_s3_key'].values[0]
        file_name = data[data['attachment_id'] == attch]['file_name'].values[0]
        textract_text_s3_link = data[data['attachment_id'] == attch]['textract_s3_link'].values[0]
        
        if verbose:
            print(f"{GRAY}│{RESET} {YELLOW}ID:{RESET}           {GREEN}{attch}{RESET}")
            print(f"{GRAY}│{RESET} {YELLOW}File Name:{RESET}    {BLUE}{file_name}{RESET}")
            print(f"{GRAY}│{RESET}")
        
        bucket_name = textract_text_s3_link.split('/')[2]
        key_name = '/'.join(textract_text_s3_link.split('/')[3:])
        response = s3.get_object(Bucket=bucket_name, Key=key_name)
        textract_data = json.loads(response['Body'].read().decode('utf-8'))
        text = get_texts_from_textract_outputs([textract_data])[0]
        data.loc[data['attachment_id'] == attch, 'text'] = text

        if verbose:
            print(f"{GRAY}│{RESET} {YELLOW}📄 Document S3:{RESET}")
            print(f"{GRAY}│{RESET}    {GRAY}s3://{document_s3_bucket}/{document_s3_key}{RESET}")
            print(f"{GRAY}│{RESET}")
            print(f"{GRAY}│{RESET} {YELLOW}📝 Textract Output:{RESET}")
            print(f"{GRAY}│{RESET}    {GRAY}{textract_text_s3_link}{RESET}")
            print(f"{GRAY}│{RESET}")
            print(f"{GRAY}│{RESET} {YELLOW}📊 Text Length:{RESET}  {MAGENTA}{len(text)} characters{RESET}")
            print(f"{GRAY}│{RESET}")
    
        # download the pdf file from s3
        response_pdf = s3.get_object(Bucket=document_s3_bucket, Key=document_s3_key)
        if pdf_download and pdf_download_dir:
            egvp_id_current_attachment = data[data['attachment_id'] == attch]['egvp_id'].values[0]
            name = egvp_id_current_attachment + "_" + str(attch) + ".pdf"
            pdf_download_path = os.path.join(pdf_download_dir, name)
            pdf_content = response_pdf['Body'].read()
            with open(pdf_download_path, 'wb') as pdf_file:
                pdf_file.write(pdf_content)
            if verbose:
                print(f"{GRAY}│{RESET} {GREEN}📥 PDF Downloaded:{RESET} {pdf_download_path}")
                print(f"{GRAY}│{RESET}")
        
        # Predictions
        if verbose:
            print(f"{GRAY}│{RESET} {CYAN}{BOLD}🤖 PREDICTIONS:{RESET}")
        attch_preds = data[data['attachment_id'] == attch]
        class_pred_type = attch_preds[(attch_preds['subtype'] == "class_pred") & (attch_preds['value'] == "'True'")]['type']
        
        prob_data = attch_preds[(attch_preds['subtype'] == "class_prob")][['type','value']]
        if prob_data.empty:
            prob_ladung = "N/A"
            prob_pfub = "N/A"
        else:
            prob_ladung = prob_data[prob_data['type']=="aftercourt_classification_ladung"]['value'].values[0]
            prob_pfub = prob_data[prob_data['type']=="aftercourt_classification_pfub"]['value'].values[0]
        
        if class_pred_type.empty:
            if verbose:
                print(f"{GRAY}│{RESET}    {RED}⚠️  NOT LADUNG OR PFUB! {RESET}")
                print(f"{GRAY}│{RESET}    {YELLOW}Prob Ladung:{RESET} {MAGENTA}{prob_ladung}{RESET}")
                print(f"{GRAY}│{RESET}    {YELLOW}Prob Pfub:{RESET}   {MAGENTA}{prob_pfub}{RESET}")
        else:
            pred = None
            if 'ladung' in class_pred_type.values[0]:
                pred = 'ladung'
            elif 'pfub' in class_pred_type.values[0]:
                pred = 'pfub'
            if verbose:
                print(f"{GRAY}│{RESET}    {YELLOW}Class:{RESET}      {GREEN}{BOLD}{pred}{RESET}")
                print(f"{GRAY}│{RESET}    {YELLOW}Prob Ladung:{RESET} {MAGENTA}{prob_ladung}{RESET}")
                print(f"{GRAY}│{RESET}    {YELLOW}Prob Pfub:{RESET}   {MAGENTA}{prob_pfub}{RESET}")
            
        
        if verbose:
            print(f"{GRAY}└{'─' * 98}{RESET}\n")
    
    
    return data[[col for col in cols if col in data.columns] + ['text']]  # Ensure only existing columns are selected


def get_data_by_ticket_uuid(ticket_uuid: str, analytics_db, s3, pdf_download: bool = False, pdf_download_dir: str = None) -> pd.DataFrame:
    query = f"""
        SELECT 
            llm_tickets.ticket_uuid,
            llm_tickets.source_type,
            llm_tickets.egvp_id,
            llm_tickets.status,
            llm_tickets.origin,
            llm_attachments.attachment_id,
            llm_attachments.s3_key as document_s3_key,
            llm_attachments.s3_bucket as document_s3_bucket,
            llm_attachments.file_name,
            llm_attachments_predictions.model_name,
            llm_attachments_predictions.type,
            llm_attachments_predictions.subtype,
            llm_attachments_predictions.value,
            textract_jobs.job_id AS textract_job_id,
            textract_jobs.status AS textract_status,
            textract_jobs.s3_link AS textract_s3_link
        FROM llm_tickets
        JOIN llm_attachments ON llm_tickets.ticket_uuid = llm_attachments.ticket_uuid
        JOIN llm_attachments_predictions ON llm_attachments.attachment_id = llm_attachments_predictions.attachment_id
        JOIN textract_jobs ON llm_attachments.attachment_id = textract_jobs.attachment_id
        WHERE llm_tickets.ticket_uuid = '{ticket_uuid}'
        ORDER BY llm_tickets.created_at DESC
    """
    data = analytics_db.sql_to_df(query)
    if data.empty:
        print(f"No data found for ticket_uuid: {ticket_uuid}")
        return data
    cols = [
        'ticket_uuid', 'source_type', 'egvp_id', 'status', 'origin',
        'attachment_id', 'document_s3_key', 'document_s3_bucket', 'file_name', 'model_name', 'type', 'subtype', 'value',
        'textract_job_id', 'textract_status', 'textract_s3_link'
    ]
    
    document_s3_bucket = data['document_s3_bucket'][0]
    document_s3_key = data['document_s3_key'][0]
    textract_text_s3_link = data['textract_s3_link'][0]
    

    bucket_name = textract_text_s3_link.split('/')[2]
    key_name = '/'.join(textract_text_s3_link.split('/')[3:])
    response = s3.get_object(Bucket=bucket_name, Key=key_name)
    textract_data = json.loads(response['Body'].read().decode('utf-8'))
    text = get_texts_from_textract_outputs([textract_data])[0]
    data['text'] = text
    
    print(f"📄 Document S3 URL: s3://{document_s3_bucket}/{document_s3_key}")
    print(f"📝 Textract S3 Link: {textract_text_s3_link}")
    

    if pdf_download and pdf_download_dir:
        # download the pdf file from s3
        response_pdf = s3.get_object(Bucket=document_s3_bucket, Key=document_s3_key)
        pdf_download_path = os.path.join(pdf_download_dir, data['ticket_uuid'][0] + ".pdf")
        pdf_content = response_pdf['Body'].read()
        with open(pdf_download_path, 'wb') as pdf_file:
            pdf_file.write(pdf_content)
        print(f"📥 PDF downloaded to: {pdf_download_path}")
    
    
    return data[[col for col in cols if col in data.columns] + ['text']]  # Ensure only existing columns are selected



def get_data_by_attachment_id(attachment_id: str, analytics_db, s3, pdf_download: bool = False, pdf_download_dir: str = None, verbose: bool = True) -> pd.DataFrame:
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    query = f"""
        SELECT 
            llm_attachments.attachment_id,
            llm_attachments.s3_key as document_s3_key,
            llm_attachments.s3_bucket as document_s3_bucket,
            llm_attachments.file_name,
            llm_attachments_predictions.model_name,
            llm_attachments_predictions.type,
            llm_attachments_predictions.subtype,
            llm_attachments_predictions.value,
            textract_jobs.job_id AS textract_job_id,
            textract_jobs.status AS textract_status,
            textract_jobs.s3_link AS textract_s3_link
        FROM llm_attachments
        JOIN llm_attachments_predictions ON llm_attachments.attachment_id = llm_attachments_predictions.attachment_id
        JOIN textract_jobs ON llm_attachments.attachment_id = textract_jobs.attachment_id
        WHERE llm_attachments.attachment_id = '{attachment_id}'
    """
    data = analytics_db.sql_to_df(query)
    if data.empty:
        print(f"No data found for attachment_id: {attachment_id}")
        return data

    cols = [
        'attachment_id', 'document_s3_key', 'document_s3_bucket', 'file_name', 'model_name', 'type', 'subtype', 'value',
        'textract_job_id', 'textract_status', 'textract_s3_link'
    ]

    if verbose:
        print(f"\n{CYAN}{BOLD}{'='*100}{RESET}")
        print(f"{CYAN}{BOLD}📎 ATTACHMENT SUMMARY{RESET}")
        print(f"{CYAN}{BOLD}{'='*100}{RESET}")
        print(f"{YELLOW}Attachment ID:{RESET}   {GREEN}{attachment_id}{RESET}")
        print(f"{YELLOW}File Name:{RESET}       {BLUE}{data['file_name'].iloc[0]}{RESET}")
        print(f"{CYAN}{BOLD}{'='*100}{RESET}\n")

    document_s3_bucket = data['document_s3_bucket'].iloc[0]
    document_s3_key = data['document_s3_key'].iloc[0]
    textract_text_s3_link = data['textract_s3_link'].iloc[0]

    bucket_name = textract_text_s3_link.split('/')[2]
    key_name = '/'.join(textract_text_s3_link.split('/')[3:])
    response = s3.get_object(Bucket=bucket_name, Key=key_name)
    textract_data = json.loads(response['Body'].read().decode('utf-8'))
    text = get_texts_from_textract_outputs([textract_data])[0]
    data['text'] = text

    if verbose:
        print(f"{GRAY}│{RESET} {YELLOW}📄 Document S3:{RESET}")
        print(f"{GRAY}│{RESET}    {GRAY}s3://{document_s3_bucket}/{document_s3_key}{RESET}")
        print(f"{GRAY}│{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}📝 Textract Output:{RESET}")
        print(f"{GRAY}│{RESET}    {GRAY}{textract_text_s3_link}{RESET}")
        print(f"{GRAY}│{RESET}")
        print(f"{GRAY}│{RESET} {YELLOW}📊 Text Length:{RESET}  {MAGENTA}{len(text)} characters{RESET}")
        print(f"{GRAY}│{RESET}")

    if pdf_download and pdf_download_dir:
        response_pdf = s3.get_object(Bucket=document_s3_bucket, Key=document_s3_key)
        name = str(attachment_id) + ".pdf"
        pdf_download_path = os.path.join(pdf_download_dir, name)
        pdf_content = response_pdf['Body'].read()
        with open(pdf_download_path, 'wb') as pdf_file:
            pdf_file.write(pdf_content)
        if verbose:
            print(f"{GRAY}│{RESET} {GREEN}📥 PDF Downloaded:{RESET} {pdf_download_path}")
            print(f"{GRAY}│{RESET}")

    # Predictions
    if verbose:
        print(f"{GRAY}│{RESET} {CYAN}{BOLD}🤖 PREDICTIONS:{RESET}")

    class_pred_type = data[(data['subtype'] == "class_pred") & (data['value'] == "'True'")]['type']

    prob_data = data[data['subtype'] == "class_prob"][['type', 'value']]
    if prob_data.empty:
        prob_ladung = "N/A"
        prob_pfub = "N/A"
    else:
        prob_ladung = prob_data[prob_data['type'] == "aftercourt_classification_ladung"]['value'].values[0] if not prob_data[prob_data['type'] == "aftercourt_classification_ladung"].empty else "N/A"
        prob_pfub = prob_data[prob_data['type'] == "aftercourt_classification_pfub"]['value'].values[0] if not prob_data[prob_data['type'] == "aftercourt_classification_pfub"].empty else "N/A"

    if class_pred_type.empty:
        if verbose:
            print(f"{GRAY}│{RESET}    {RED}⚠️  NOT LADUNG OR PFUB! {RESET}")
            print(f"{GRAY}│{RESET}    {YELLOW}Prob Ladung:{RESET} {MAGENTA}{prob_ladung}{RESET}")
            print(f"{GRAY}│{RESET}    {YELLOW}Prob Pfub:{RESET}   {MAGENTA}{prob_pfub}{RESET}")
    else:
        pred = None
        if 'ladung' in class_pred_type.values[0]:
            pred = 'ladung'
        elif 'pfub' in class_pred_type.values[0]:
            pred = 'pfub'
        if verbose:
            print(f"{GRAY}│{RESET}    {YELLOW}Class:{RESET}      {GREEN}{BOLD}{pred}{RESET}")
            print(f"{GRAY}│{RESET}    {YELLOW}Prob Ladung:{RESET} {MAGENTA}{prob_ladung}{RESET}")
            print(f"{GRAY}│{RESET}    {YELLOW}Prob Pfub:{RESET}   {MAGENTA}{prob_pfub}{RESET}")

    if verbose:
        print(f"{GRAY}└{'─' * 98}{RESET}\n")

    return data[[col for col in cols if col in data.columns] + ['text']]


def download_pdf_by_document_s3_info(s3, document_s3_bucket: str, document_s3_key: str, download_dir: str, file_name: str = None) -> str:
    """
    Download a PDF file from S3 using document_s3_bucket and document_s3_key
    (as returned by get_data_by_egvp_id, get_data_by_ticket_uuid, etc.).

    Args:
        s3: boto3 S3 client
        document_s3_bucket: S3 bucket name (from the document_s3_bucket column)
        document_s3_key: S3 object key (from the document_s3_key column)
        download_dir: Local directory to save the PDF
        file_name: Optional custom file name. If None, uses the object key's file name

    Returns:
        str: Path to the downloaded PDF file
    """
    if file_name is None:
        file_name = os.path.basename(document_s3_key)

    if not file_name.endswith('.pdf'):
        file_name += '.pdf'

    os.makedirs(download_dir, exist_ok=True)
    download_path = os.path.join(download_dir, file_name)

    try:
        response = s3.get_object(Bucket=document_s3_bucket, Key=document_s3_key)
        pdf_content = response['Body'].read()

        with open(download_path, 'wb') as pdf_file:
            pdf_file.write(pdf_content)

        print(f"✅ PDF downloaded successfully: {download_path}")
        return download_path

    except Exception as e:
        print(f"❌ Error downloading PDF: {str(e)}")
        raise


def download_pdf_from_s3(s3, bucket_name: str, object_key: str, download_dir: str, file_name: str = None) -> str:
    """
    Download a PDF file from S3 given the bucket name and object key.
    
    Args:
        s3: boto3 S3 client
        bucket_name: S3 bucket name
        object_key: S3 object key (path to the file in the bucket)
        download_dir: Local directory to save the PDF
        file_name: Optional custom file name. If None, uses the object key's file name
    
    Returns:
        str: Path to the downloaded PDF file
    """
    if file_name is None:
        file_name = os.path.basename(object_key)
    
    # Ensure the file has .pdf extension
    if not file_name.endswith('.pdf'):
        file_name += '.pdf'
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    download_path = os.path.join(download_dir, file_name)
    
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        pdf_content = response['Body'].read()
        
        with open(download_path, 'wb') as pdf_file:
            pdf_file.write(pdf_content)
        
        print(f"✅ PDF downloaded successfully: {download_path}")
        return download_path
    
    except Exception as e:
        print(f"❌ Error downloading PDF: {str(e)}")
        raise
