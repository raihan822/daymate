# x= 5
# prompt = f"""User is at {x}'.
# Weather: {x}, temp {x}°C.
# Top headlines: {x}.
# Generate a concise daily plan (3-6 items) and practical recommendations (carry items, suggest reschedule if needed)."""
# print(type(prompt))
# print(prompt)


def generate_plan(req: dict = None, weather_info: str = "", news_info: str = ""):
    if weather_info and news_info:
        print("both weather and news are present.")
    elif req is not None:
        print("Req is present, and either weather or news maybe missing.")
    else:
        print("either Req is absent and either weather or news is missing.")


generate_plan(req={3:15}, weather_info="hi",news_info="raihan")
generate_plan(req={3:15})
generate_plan(weather_info="hi",news_info="raihan")
generate_plan(req={3:15}, weather_info="hi")
generate_plan()
