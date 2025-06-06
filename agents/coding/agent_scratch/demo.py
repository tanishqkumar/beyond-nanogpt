#!/usr/bin/env python3
"""
🐾 Digital Pet Care System Demo 🐾
A cute and interactive virtual pet management system!

Author: Step Agent
Version: 1.0
"""

import time
import random
from pet import Pet
from care_system import PetCareSystem
from utils import (
    get_random_pet_fact, get_caring_tip, format_pet_grid,
    get_time_greeting, calculate_care_score, get_motivational_message
)

def print_header():
    """Print a cute header for the demo! 🎨"""
    print("\n" + "=" * 60)
    print("🐾" + " " * 15 + "DIGITAL PET CARE SYSTEM" + " " * 15 + "🐾")
    print("" + " " * 8 + "Where Virtual Pets Get All The Love!" + " " * 8 + "")
    print("=" * 60 + "\n")

def print_separator():
    """Print a cute separator! ✨"""
    print("\n" + "~" * 50 + "\n")

def demo_basic_pet_creation():
    """Demonstrate basic pet creation and interaction! 🐱"""
    print("🎯 DEMO 1: Creating and Caring for Your First Pet!")
    print_separator()
    
    # Create a pet care system
    care_system = PetCareSystem()
    
    # Adopt our first pet
    print("Let's adopt our first pet! 🎉")
    result = care_system.adopt_pet("Whiskers", "cat")
    print(result)
    
    # Get our pet
    whiskers = care_system.get_pet_by_name("Whiskers")
    
    # Show initial status
    print("\n📊 Initial status:")
    print(whiskers.get_status())
    
    # Feed the pet
    print("\n🍽️ Let's feed Whiskers:")
    result = whiskers.feed()
    print(result)
    
    # Play with the pet
    print("\n🎾 Time to play!")
    result = whiskers.play()
    print(result)
    
    # Show updated status
    print("\n📊 Status after care:")
    print(whiskers.get_status())
    
    return care_system

def demo_multiple_pets(care_system):
    """Demonstrate managing multiple pets! 🐕🐰🐹"""
    print("\n🎯 DEMO 2: Managing Multiple Adorable Pets!")
    print_separator()
    
    # Adopt more pets
    pets_to_adopt = [
        ("Buddy", "dog"),
        ("Cocoa", "bunny"),
        ("Peanut", "hamster")
    ]
    
    for name, species in pets_to_adopt:
        result = care_system.adopt_pet(name, species)
        print(result)
        time.sleep(0.5)  # Small delay for dramatic effect
    
    # List all pets
    print("\n👥 Your Pet Family:")
    print(care_system.list_pets())
    
    # Show pet grid
    print(format_pet_grid(care_system.pets))
    
    # Care for all pets at once
    print("🎉 Let's play with ALL pets at once!")
    result = care_system.care_for_all_pets("play")
    print(result)
    
    print("\n🍽️ Feeding time for everyone!")
    result = care_system.care_for_all_pets("feed")
    print(result)
    
    return care_system

def demo_pet_management_features(care_system):
    """Demonstrate advanced pet management features! 📊"""
    print("\n🎯 DEMO 3: Advanced Pet Care Features!")
    print_separator()
    
    # Daily summary
    print("📈 Daily Pet Report:")
    print(care_system.get_daily_summary())
    
    # Care score calculation
    score, rating = calculate_care_score(care_system.pets)
    print(f"\n🏆 Your Care Score: {score:.1f}/100 - {rating}")
    
    # Random pet fact
    print(f"\n🧠 Fun Fact: {get_random_pet_fact()}")
    
    # Caring tip
    print(f"\n💡 Pro Tip: {get_caring_tip()}")
    
    # Motivational message
    print(f"\n💪 {get_motivational_message()}")
    
    # Save pets
    print("\n💾 Saving your pets to file...")
    result = care_system.save_pets()
    print(result)

def demo_individual_pet_care(care_system):
    """Demonstrate individual pet care! 🎯"""
    print("\n🎯 DEMO 4: Individual Pet Care & Status Checking!")
    print_separator()
    
    # Pick a random pet to focus on
    if care_system.pets:
        target_pet = random.choice(care_system.pets)
        print(f"Let's focus on {target_pet.name} the {target_pet.species}! 🎯")
        
        # Show detailed status
        print(care_system.care_for_pet(target_pet.name, "status"))
        
        # Let them rest if they're tired
        if target_pet.energy < 50:
            print(f"\n😴 {target_pet.name} looks tired, let's help them rest:")
            result = care_system.care_for_pet(target_pet.name, "rest")
            print(result)
        
        # Play if they have energy
        if target_pet.energy > 30:
            print(f"\n🎾 {target_pet.name} has energy - let's play!")
            result = care_system.care_for_pet(target_pet.name, "play")
            print(result)
        
        # Show final status
        print(f"\n📊 {target_pet.name}'s updated status:")
        print(care_system.care_for_pet(target_pet.name, "status"))

def demo_random_name_generator(care_system):
    """Demonstrate the random name generator! 🎲"""
    print("\n🎯 DEMO 5: Random Pet Name Generator!")
    print_separator()
    
    print("🎲 Let's adopt some pets with random names and species!")
    
    for i in range(3):
        random_name = care_system.generate_pet_name()
        result = care_system.adopt_pet(random_name)  # Random species too!
        print(result)
        time.sleep(0.3)
    
    print("\n🌟 Look at all these wonderful pets!")
    print(care_system.list_pets())

def main():
    """Run the complete demo! 🚀"""
    print_header()
    
    # Time-based greeting
    print(get_time_greeting())
    
    try:
        # Run all demo sections
        care_system = demo_basic_pet_creation()
        
        input("\n🎮 Press Enter to continue to the next demo...")  # Optional pause
        care_system = demo_multiple_pets(care_system)
        
        input("\n🎮 Press Enter to continue to the next demo...")  # Optional pause
        demo_pet_management_features(care_system)
        
        input("\n🎮 Press Enter to continue to the next demo...")  # Optional pause
        demo_individual_pet_care(care_system)
        
        input("\n🎮 Press Enter for the final demo...")  # Optional pause
        demo_random_name_generator(care_system)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎊 DEMO COMPLETE! 🎊")
        print("=" * 60)
        
        print("\n🌟 Final Pet Family Summary:")
        print(care_system.list_pets())
        
        # Final care score
        score, rating = calculate_care_score(care_system.pets)
        print(f"\n🏆 Final Care Score: {score:.1f}/100 - {rating}")
        
        print(f"\n🎉 Thanks for experiencing the Digital Pet Care System!")
        print(f"💖 You've successfully cared for {len(care_system.pets)} adorable virtual pets!")
        print(f"\n{get_motivational_message()}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for trying the Digital Pet Care System! Come back anytime!")
    except Exception as e:
        print(f"\n❌ Oops! Something went wrong: {e}")
        print("🔧 Don't worry, your virtual pets are safe!")

if __name__ == "__main__":
    main()