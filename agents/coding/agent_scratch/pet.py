import random
import datetime

class Pet:
    """A adorable virtual pet that needs love and care! 🐾"""
    
    def __init__(self, name, species="cat"):
        self.name = name
        self.species = species
        self.happiness = 50
        self.hunger = 30
        self.energy = 70
        self.age = 0
        self.created_at = datetime.datetime.now()
        self.last_fed = None
        self.last_played = None
        
    def feed(self):
        """Feed your pet to reduce hunger and increase happiness! 🍽️"""
        if self.hunger <= 10:
            return f"{self.name} is already full! Maybe try playing instead? 😊"
        
        self.hunger = max(0, self.hunger - 25)
        self.happiness = min(100, self.happiness + 15)
        self.last_fed = datetime.datetime.now()
        
        food_reactions = [
            f"{self.name} munches happily! *nom nom nom* 🍗",
            f"{self.name} purrs with delight! So tasty! 😋",
            f"{self.name} wags tail excitedly while eating! 🐕",
            f"{self.name} does a little happy dance! Food is the best! 💃"
        ]
        return random.choice(food_reactions)
    
    def play(self):
        """Play with your pet to increase happiness and reduce energy! 🎾"""
        if self.energy <= 10:
            return f"{self.name} is too tired to play! Let them rest first 😴"
        
        self.energy = max(0, self.energy - 20)
        self.happiness = min(100, self.happiness + 25)
        self.last_played = datetime.datetime.now()
        
        play_activities = [
            f"{self.name} chases their tail in circles! So fun! 🌪️",
            f"{self.name} plays fetch and brings back a stick! Good pet! 🎾",
            f"{self.name} does adorable tricks for treats! *bows* 🎪",
            f"{self.name} zooms around the room with pure joy! ⚡",
            f"{self.name} pounces on imaginary butterflies! So cute! 🦋"
        ]
        return random.choice(play_activities)
    
    def rest(self):
        """Let your pet rest to restore energy! 💤"""
        if self.energy >= 90:
            return f"{self.name} is already well-rested and ready for adventure! 🌟"
        
        self.energy = min(100, self.energy + 30)
        rest_messages = [
            f"{self.name} curls up in a cozy ball and naps peacefully 😴",
            f"{self.name} stretches out in a sunny spot for a power nap ☀️",
            f"{self.name} dreams of chasing mice and playing! *soft snores* 💭"
        ]
        return random.choice(rest_messages)
    
    def get_mood(self):
        """Get your pet's current mood based on their stats! 😊"""
        if self.happiness >= 80:
            return "ecstatic 🌟"
        elif self.happiness >= 60:
            return "happy 😊"
        elif self.happiness >= 40:
            return "content 🙂"
        elif self.happiness >= 20:
            return "a bit sad 😔"
        else:
            return "very unhappy 😢"
    
    def get_status(self):
        """Get a cute status report of your pet! 📊"""
        age_days = (datetime.datetime.now() - self.created_at).days
        
        status = f"\n🐾 === {self.name} the {self.species} === 🐾\n"
        status += f"Age: {age_days} days old\n"
        status += f"Mood: {self.get_mood()}\n"
        status += f"Happiness: {self.happiness}/100 {'❤️' * (self.happiness // 20)}\n"
        status += f"Hunger: {self.hunger}/100 {'🍽️' * (self.hunger // 20)}\n"
        status += f"Energy: {self.energy}/100 {'⚡' * (self.energy // 20)}\n"
        
        if self.last_fed:
            time_since_fed = datetime.datetime.now() - self.last_fed
            status += f"Last fed: {time_since_fed.seconds // 3600} hours ago\n"
        
        if self.last_played:
            time_since_played = datetime.datetime.now() - self.last_played
            status += f"Last played: {time_since_played.seconds // 3600} hours ago\n"
        
        return status
    
    def age_up(self):
        """Age your pet (happens automatically over time) 🎂"""
        self.age += 1
        # Pets get a bit hungrier and less energetic as time passes
        self.hunger = min(100, self.hunger + 5)
        self.energy = max(0, self.energy - 3)
        
        if self.age % 10 == 0:  # Special birthday message every 10 age units
            return f"🎉 Happy Birthday {self.name}! You're getting wiser! 🎂"
        return None