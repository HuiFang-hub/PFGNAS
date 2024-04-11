import google.generativeai as genai

if __name__ == '__main__':
    GOOGLE_API_KEY= 'AIzaSyBQ8F6cyCvh9DkbUMrYfdfr-DQwoVERIVo'
    genai.configure(api_key=GOOGLE_API_KEY)
    response = genai.chat(messages='Hello')
    print(response.last)
