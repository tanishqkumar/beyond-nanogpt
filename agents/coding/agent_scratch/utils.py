import random
import datetime

def get_random_pet_fact():
    """Get a fun random fact about pets! 🧠"""
    facts = [
        "🐱 Cats spend 70% of their lives sleeping - that's 13-16 hours a day!",
        "🐕 Dogs have a sense of smell 40 times better than humans!",
        "🐰 Rabbits can jump nearly 3 feet high and 9 feet long!",
        "🐹 Hamsters can store food in their cheek pouches!",
        "🦜 Parrots can live over 100 years!",
        "🐢 Turtles have been around for over 200 million years!",
        "🐠 Goldfish can remember things for months, not just seconds!",
        "🐕 Dogs can learn over 150 words and count to 4 or 5!",
        "🐱 A group of cats is called a 'clowder'!",
        "🐰 Rabbits can see behind them without turning their heads!"
    ]
    return random.choice(facts)

def get_caring_tip():
    """Get a helpful tip for caring for virtual pets! 💡"""
    tips = [
        "🎾 Playing with your pets increases their happiness the most!",
        "🍽️ Don't overfeed - pets won't eat if they're already full!",
        "😴 Let tired pets rest to restore their energy!",
        "📊 Check your pet's status regularly to see how they're doing!",
        "👥 Adopt multiple pets - they love having friends!",
        "⏰ Visit your pets regularly - they miss you when you're gone!",
        "🎉 Try caring for all your pets at once for efficiency!",
        "📈 Happy pets are healthy pets - keep that happiness up!",
        "🏠 Create a loving virtual home for all your pets!",
        "💖 The more love you give, the more love you get back!"
    ]
    return random.choice(tips)

def format_pet_grid(pets):
    """Format pets in a nice grid layout for display! 📋"""
    if not pets:
        return "No pets to display! 🐾"
    
    grid = "\n🌟 === Pet Overview Grid === 🌟\n"
    grid += "-" * 50 + "\n"
    
    for i, pet in enumerate(pets, 1):
        happiness_bar = "❤️" * (pet.happiness // 20)
        energy_bar = "⚡" * (pet.energy // 20)
        hunger_bar = "🍽️" * (pet.hunger // 20)
        
        grid += f"{i:2d}. {pet.name:<12} ({pet.species:<10}) | "
        grid += f"😊{happiness_bar:<5} ⚡{energy_bar:<5} 🍽️{hunger_bar:<5}\n"
    
    grid += "-" * 50 + "\n"
    return grid

def get_time_greeting():
    """Get a time-appropriate greeting! 🌅"""
    current_hour = datetime.datetime.now().hour
    
    if 5 <= current_hour < 12:
        greetings = [
            "🌅 Good morning! Your pets are excited to see you!",
            "☀️ Rise and shine! Time to check on your furry friends!",
            "🐓 Morning! Your pets have been waiting patiently for you!"
        ]
    elif 12 <= current_hour < 17:
        greetings = [
            "🌞 Good afternoon! Perfect time for some pet playtime!",
            "🌤️ Afternoon check-in! How are your pets doing?",
            "⏰ Midday pet care time! They're ready for attention!"
        ]
    elif 17 <= current_hour < 21:
        greetings = [
            "🌆 Good evening! Time to wind down with your pets!",
            "🌇 Evening cuddle time with your virtual companions!",
            "🏡 Welcome home! Your pets missed you today!"
        ]
    else:
        greetings = [
            "🌙 Good night! Don't forget to check on your sleepy pets!",
            "⭐ Late night pet care! They appreciate your dedication!",
            "🌜 Nighttime snuggles with your virtual friends!"
        ]
    
    return random.choice(greetings)

def calculate_care_score(pets):
    """Calculate how well you're caring for all your pets! 🏆"""
    if not pets:
        return 0, "No pets to care for yet!"
    
    total_score = 0
    for pet in pets:
        # Happiness is most important (50% weight)
        happiness_score = pet.happiness * 0.5
        # Energy and hunger balance (25% each)
        energy_score = pet.energy * 0.25
        hunger_score = (100 - pet.hunger) * 0.25  # Lower hunger is better
        
        pet_score = happiness_score + energy_score + hunger_score
        total_score += pet_score
    
    average_score = total_score / len(pets)
    
    if average_score >= 85:
        rating = "🏆 Excellent Pet Parent!"
    elif average_score >= 70:
        rating = "🥈 Great Pet Caregiver!"
    elif average_score >= 55:
        rating = "🥉 Good Pet Friend!"
    elif average_score >= 40:
        rating = "📈 Learning Pet Care!"
    else:
        rating = "💪 Keep Trying!"
    
    return average_score, rating

def get_motivational_message():
    """Get an encouraging message for pet care! 💪"""
    messages = [
        "🌟 Every moment with your pets is precious!",
        "💝 Your love makes their virtual world brighter!",
        "🚀 Keep up the amazing pet care work!",
        "🌈 You're creating joy for your digital companions!",
        "⭐ Your pets are lucky to have such a caring owner!",
        "🎯 Small acts of care make a big difference!",
        "💖 The bond with your pets grows stronger every day!",
        "🌺 Your dedication to pet care is inspiring!",
        "🎊 Celebrate every happy moment with your pets!",
        "🔥 You're becoming a pet care expert!"
    ]
    return random.choice(messages)