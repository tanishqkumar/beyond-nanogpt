import json
import random
from pet import Pet

class PetCareSystem:
    """A magical pet care system to manage all your adorable virtual pets! 🏠"""
    
    def __init__(self):
        self.pets = []
        self.pet_types = [
            "cat", "dog", "bunny", "hamster", "parrot", 
            "turtle", "goldfish", "ferret", "chinchilla"
        ]
        
    def adopt_pet(self, name, species=None):
        """Adopt a new pet! So exciting! 🎉"""
        if species is None:
            species = random.choice(self.pet_types)
        
        new_pet = Pet(name, species)
        self.pets.append(new_pet)
        
        welcome_messages = [
            f"🎊 Congratulations! You've adopted {name} the {species}!",
            f"🌟 Welcome home, {name}! Your new {species} is so happy to meet you!",
            f"💕 {name} the {species} has found their forever home with you!",
            f"🎈 A new adventure begins with {name} your adorable {species}!"
        ]
        return random.choice(welcome_messages)
    
    def list_pets(self):
        """See all your wonderful pets! 👀"""
        if not self.pets:
            return "💔 You don't have any pets yet! Why not adopt one? 🐾"
        
        pet_list = "\n🏠 === Your Pet Family === 🏠\n"
        for i, pet in enumerate(self.pets, 1):
            pet_list += f"{i}. {pet.name} the {pet.species} - {pet.get_mood()}\n"
        
        return pet_list
    
    def get_pet_by_name(self, name):
        """Find a pet by their name! 🔍"""
        for pet in self.pets:
            if pet.name.lower() == name.lower():
                return pet
        return None
    
    def get_pet_by_index(self, index):
        """Get a pet by their position in the list! 📋"""
        if 0 <= index < len(self.pets):
            return self.pets[index]
        return None
    
    def care_for_pet(self, pet_identifier, action):
        """Take care of a specific pet! 💖"""
        # Try to get pet by name first, then by index
        pet = None
        if isinstance(pet_identifier, str):
            pet = self.get_pet_by_name(pet_identifier)
        elif isinstance(pet_identifier, int):
            pet = self.get_pet_by_index(pet_identifier - 1)  # 1-indexed for user
        
        if not pet:
            return "😞 Couldn't find that pet! Check the name or number and try again."
        
        if action.lower() == "feed":
            return pet.feed()
        elif action.lower() == "play":
            return pet.play()
        elif action.lower() == "rest":
            return pet.rest()
        elif action.lower() == "status":
            return pet.get_status()
        else:
            return "🤔 I don't understand that action! Try: feed, play, rest, or status"
    
    def care_for_all_pets(self, action):
        """Take care of ALL your pets at once! 🌟"""
        if not self.pets:
            return "💔 You don't have any pets to care for yet!"
        
        results = []
        for pet in self.pets:
            if action.lower() == "feed":
                result = pet.feed()
            elif action.lower() == "play":
                result = pet.play()
            elif action.lower() == "rest":
                result = pet.rest()
            else:
                continue
            results.append(f"{pet.name}: {result}")
        
        return "\n".join(results)
    
    def get_daily_summary(self):
        """Get a summary of all your pets' wellbeing! 📈"""
        if not self.pets:
            return "📭 No pets to summarize! Consider adopting one? 🐾"
        
        summary = "\n📊 === Daily Pet Report === 📊\n"
        
        total_happiness = sum(pet.happiness for pet in self.pets)
        avg_happiness = total_happiness / len(self.pets)
        
        summary += f"Overall Family Happiness: {avg_happiness:.1f}/100\n"
        summary += f"Total Pets: {len(self.pets)}\n\n"
        
        # Categorize pets by mood
        happy_pets = [pet for pet in self.pets if pet.happiness >= 70]
        sad_pets = [pet for pet in self.pets if pet.happiness < 40]
        
        if happy_pets:
            summary += f"😊 Happy Pets ({len(happy_pets)}): "
            summary += ", ".join([pet.name for pet in happy_pets]) + "\n"
        
        if sad_pets:
            summary += f"😔 Pets Needing Attention ({len(sad_pets)}): "
            summary += ", ".join([pet.name for pet in sad_pets]) + "\n"
        
        return summary
    
    def save_pets(self, filename="./agent_scratch/pets_save.json"):
        """Save your pets to a file! 💾"""
        try:
            pets_data = []
            for pet in self.pets:
                pet_data = {
                    "name": pet.name,
                    "species": pet.species,
                    "happiness": pet.happiness,
                    "hunger": pet.hunger,
                    "energy": pet.energy,
                    "age": pet.age,
                    "created_at": pet.created_at.isoformat()
                }
                pets_data.append(pet_data)
            
            with open(filename, 'w') as f:
                json.dump(pets_data, f, indent=2)
            
            return f"💾 Successfully saved {len(self.pets)} pets to {filename}!"
        except Exception as e:
            return f"😞 Error saving pets: {str(e)}"
    
    def generate_pet_name(self):
        """Generate a random cute pet name! 🎲"""
        cute_names = [
            "Whiskers", "Fluffy", "Cuddles", "Snuggles", "Peanut",
            "Bubbles", "Mochi", "Cookie", "Marshmallow", "Honey",
            "Ziggy", "Luna", "Ollie", "Bella", "Max", "Coco",
            "Pepper", "Ginger", "Oreo", "Pickles", "Waffles"
        ]
        return random.choice(cute_names)