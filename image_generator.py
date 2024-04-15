from openai import OpenAI

def generate_image(prompt):
    client = OpenAI()

    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    image_url = response.data[0].url

    return image_url

def generate_prompt(tags):
    client = OpenAI()

    instructions = """To generate a delightful 3d render style picture of a room by using an AI image generation, descriptive prompts are needed, a good prompt will be like 
                        \" An eclectic room filled with various items: a wooden desk covered with a computer setup, including multiple monitors, a keyboard, and a mouse. Nearby, a laptop rests on a small table. The room features anime figurines scattered across the desk, a bookshelf filled with books, and a dresser. A wooden chair and a bag chair provide seating options. A fan is visible near the ceiling, and cords are strewn about. The walls are adorned with framed pictures and the floor is white. Additional items include a stuffed animal, a holder for toothpaste, and several other electronic devices. \" 
                        here are the keywords, you should use these keywords to create a prompt. Avoid making image messy. Please output the prompt *directly*."""
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "system", "content": f"{instructions}"},
                {"role": "user", "content": f"{tags}"}])
    
    return response.choices[0].message.content