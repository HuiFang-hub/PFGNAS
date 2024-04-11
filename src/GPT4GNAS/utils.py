def init_llm(api_key):

    headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}

    return headers
