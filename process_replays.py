import pandas as pd
import re
import copy
import os
import argparse
import glob
import json

# --- Helper functions ---

# --- NEW: Normalization Function ---
# Common battle-only forme suffixes that should be stripped for matching
FORME_SUFFIXES = [
    # Battle-only temporary formes
    'busted', 'hangry', 'hero', 'crowned', 'origin', 'sky', 'primal',
    'complete', 'tenpc', '10', '50', 'school', 'meteor', 'core',
    'blade', 'shield', 'ultra', 'dusk', 'dawn', 'altered',
    # Tera formes
    'terastal', 'tera', 'stellar',
    # Size/form variants often shown differently
    'small', 'large', 'super', 'average', 'plumage',
    # Therian/Incarnate formes (keep therian as base since it's common)
    'incarnate',
    # Segment variants
    'threesegment', 'twosegment',
    # Family size
    'familyofthree', 'familyoffour', 'three', 'four',
    # Colors (Squawkabilly, etc.)
    'green', 'blue', 'yellow', 'white',
    # Tatsugiri
    'curly', 'droopy', 'stretchy',
    # Ogerpon
    'wellspring', 'hearthflame', 'cornerstone', 'teal',
    # Lycanroc
    'midday', 'midnight',
    # Toxtricity
    'amped', 'lowkey',
    # Urshifu
    'rapidstrike', 'singlestrike',
    # Wormadam
    'sandy', 'trash', 'plant',
    # Basculin
    'bluestriped', 'redstriped', 'whitestriped',
    # Rotom forms - keep as distinct since they have different types/moves
    # 'wash', 'heat', 'frost', 'fan', 'mow',
    # Gender suffixes for species with gender differences
    'female', 'male', 'f', 'm',
]

def normalize_species_name(species_str, strip_formes=True):
    """Converts species name to lowercase, removing spaces and hyphens.
    If strip_formes=True, also attempts to strip common forme suffixes for matching."""
    if not isinstance(species_str, str):
        return 'unknown'
    
    # Lowercase, remove hyphens, remove spaces
    normalized = species_str.lower().replace('-', '').replace(' ', '')
    
    if not normalized:
        return 'unknown'
    
    if not strip_formes:
        return normalized
    
    # --- Try to strip forme suffixes for matching ---
    # Sort by length descending so we strip longer suffixes first
    for suffix in sorted(FORME_SUFFIXES, key=len, reverse=True):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            base = normalized[:-len(suffix)]
            # Make sure we're not stripping too much (base should be at least 3 chars)
            if len(base) >= 3:
                return base
    
    return normalized


def normalize_move_name(move_str):
    """Normalize move names the same way as species (lowercase, no spaces/hyphens).
    FIX #7: Ensures move name consistency across dataset."""
    if not isinstance(move_str, str):
        return 'unknown'
    
    # Remove special tags first
    cleaned = re.sub(r'\[from\].*|\[still\]', '', move_str).strip()
    
    # Normalize: lowercase, no hyphens, no spaces
    normalized = cleaned.lower().replace('-', '').replace(' ', '')
    
    return normalized if normalized else 'unknown'


def get_base_species(normalized_name):
    """Attempt to get base species by stripping forme suffixes.
    Returns a list of possible base names to try, most specific first."""
    candidates = [normalized_name]  # Always include original
    
    for suffix in sorted(FORME_SUFFIXES, key=len, reverse=True):
        if normalized_name.endswith(suffix) and len(normalized_name) > len(suffix):
            base = normalized_name[:-len(suffix)]
            if len(base) >= 3 and base not in candidates:
                candidates.append(base)
    
    return candidates

def parse_hp(hp_str):
    """Parses Showdown HP strings (e.g., '100/100', '50%', 'fnt', '75', '36/100 tox') into percentages.
    FIX #2: Added validation to clamp percentages to 0-100 range."""
    if hp_str == 'fnt': return 0
    if hp_str.isdigit(): 
        # FIX #2: Clamp single digit values
        return max(0, min(100, int(hp_str)))

    try:
        # Attempt to strip status BEFORE splitting
        match_hp_with_status = re.match(r'^(\d+/\d+|\d+%?)(\s+\w+)?$', hp_str.strip())
        if match_hp_with_status:
            hp_part = match_hp_with_status.group(1)
            if '/' in hp_part:
                parts = hp_part.split('/')
                if len(parts) == 2:
                    numerator = int(parts[0])
                    denominator = int(parts[1])
                    if denominator > 0:
                        percentage = round((numerator / denominator) * 100)
                        # FIX #2: Clamp to valid range
                        return max(0, min(100, percentage))
                    return 0
            elif hp_part.endswith('%'):
                percentage = int(hp_part[:-1])
                # FIX #2: Clamp to valid range
                return max(0, min(100, percentage))
            elif hp_part.isdigit():
                percentage = int(hp_part)
                # FIX #2: Clamp to valid range
                return max(0, min(100, percentage))
            else:
                # print(f"Warning: Regex matched but failed to parse HP part: '{hp_part}' from '{hp_str}'")
                return None
        elif '/' in hp_str:
            parts = hp_str.split('/')
            if len(parts) == 2:
                numerator = int(parts[0])
                denominator = int(parts[1])
                if denominator > 0:
                    percentage = round((numerator / denominator) * 100)
                    # FIX #2: Clamp to valid range
                    return max(0, min(100, percentage))
                return 0
        elif hp_str.endswith('%'):
            percentage = int(hp_str[:-1])
            # FIX #2: Clamp to valid range
            return max(0, min(100, percentage))
        return None

    except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
        # print(f"Warning: Exception during HP parsing '{hp_str}': {e}")
        return None

def get_initial_pokemon_state():
    """Returns a dictionary representing the default state of a Pokémon slot."""
    return {
        'species': normalize_species_name('Unknown'),
        'hp_perc': 100,
        'status': 'none',
        'is_active': False,
        'is_fainted': False,
        'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0},
        'terastallized': False,
        'tera_type': 'none',
        'revealed': False,
        'revealed_moves': set()
    }

# --- Identify known conditions ---
PSEUDO_WEATHERS = {'Trick Room'}
SIDE_CONDITIONS = {
    'Reflect', 'Light Screen', 'Aurora Veil', 'Safeguard', 'Tailwind',
    'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web',
}
HAZARD_CONDITIONS = {'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web'}

def get_initial_field_state():
    """Returns a dictionary representing the default state of the battle field."""
    state = {
        'weather': 'none', 'weather_turns': 0,
        'pseudo_weather': 'none',
        'terrain': 'none', 'terrain_turns': 0,
        'p1_hazards': {hz: 0 for hz in HAZARD_CONDITIONS},
        'p2_hazards': {hz: 0 for hz in HAZARD_CONDITIONS},
        'p1_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS},
        'p2_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS},
    }
    return state

def parse_player_from_side(side_str):
    """Extracts player identifier ('p1' or 'p2') from a side identifier string."""
    match = re.match(r'(p[12])', side_str)
    return match.group(1) if match else None

def parse_condition_name(condition_str):
    """Extracts the condition name from a Showdown condition string."""
    return condition_str.split(': ')[-1].strip()

def parse_pokemon_identifier(identifier):
    """Extracts player ('p1'/'p2') and nickname/species name from a Pokémon identifier string."""
    match = re.match(r'(p[12])[a-z]?(?::\s*(.*))?', identifier)
    if match:
        player = match.group(1)
        name_part = match.group(2).strip() if match.group(2) else None
        return player, name_part
    match_player_only = re.match(r'(p[12])$', identifier)
    if match_player_only:
        return match_player_only.group(1), None
    return None, None

# --- Main Parsing Function ---
def parse_showdown_replay(log_text, replay_id="unknown"):
    """
    Parses a raw Pokemon Showdown replay log text into a list of game state dictionaries.
    """
    lines = log_text.strip().split('\n')
    game_states = []
    current_state = {
        'replay_id': replay_id,
        'turn_number': 0,
        'player_to_move': None,
        'p1': {}, 'p2': {},
        'field': get_initial_field_state(),
        'action_taken': None,
        'battle_winner': None,
        'last_move_p1': 'none',
        'last_move_p2': 'none'
    }
    player_names = {'p1': None, 'p2': None}
    # Maps slotX to the *normalized* initial species revealed in |poke| lines
    slot_id_to_species = {'p1': {}, 'p2': {}}
    nickname_to_slot_id = {'p1': {}, 'p2': {}}
    active_slot = {'p1': None, 'p2': None}
    current_teamsize = {'p1': 0, 'p2': 0}
    slot_counters = {'p1': 0, 'p2': 0}
    last_move_seen = {'p1': 'none', 'p2': 'none'}
    
    # FIX #6: Track if action was already recorded this turn (prevents multi-hit duplicates)
    action_recorded_this_turn = {'p1': False, 'p2': False}

    # --- Phase 1: Initial Setup (Teamsize, Initial Pokemon Species) ---
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]

        if event_type == 'player':
            if len(parts) >= 4 and (player_names['p1'] is None or player_names['p2'] is None):
                player_id, name = parts[2], parts[3]
                if player_id in player_names: player_names[player_id] = name.strip()

        elif event_type == 'teamsize':
            if len(parts) >= 4:
                player_id, size_str = parts[2], parts[3]
                try:
                    size = int(size_str)
                    if player_id in current_teamsize:
                        current_teamsize[player_id] = size
                        current_state[player_id] = {f'slot{i+1}': get_initial_pokemon_state() for i in range(size)}
                except ValueError:
                    print(f"Warning: Invalid teamsize value '{size_str}' in |teamsize| line: {line.strip()}")

        elif event_type == 'poke':
            if len(parts) >= 3:
                player_id, details, *_ = parts[2:]
                if player_id in slot_counters:
                    # Extract and NORMALIZE species name
                    raw_species = details.split(',')[0].split('|')[-1].replace('-*', '')
                    normalized_species = normalize_species_name(raw_species)

                    slot_counters[player_id] += 1
                    slot_id = f'slot{slot_counters[player_id]}'

                    if slot_counters[player_id] <= current_teamsize.get(player_id, 6):
                        if player_id in current_state and slot_id in current_state[player_id]:
                            # Store NORMALIZED species for lookup and in state
                            slot_id_to_species[player_id][slot_id] = normalized_species
                            current_state[player_id][slot_id]['species'] = normalized_species
                        else:
                            if player_id in current_state:
                                current_state[player_id][slot_id] = get_initial_pokemon_state()
                                current_state[player_id][slot_id]['species'] = normalized_species
                                slot_id_to_species[player_id][slot_id] = normalized_species


    # --- Slot ID Lookup Function ---
    def find_slot_id(player, identifier_name, species_from_details):
        """
        Finds the slot_id for a Pokémon based on identifier name (nickname/species)
        and the NORMALIZED species name from details.
        
        FIX #5: Now checks current state species FIRST (handles forme changes),
        then checks initial species mapping.
        
        Updates nickname mapping dynamically. Prioritizes nickname map, then exact normalized species match.
        If no match found and we have room, dynamically creates a new slot (for replays without |poke| lines).
        Returns None on failure.
        """
        nonlocal nickname_to_slot_id, slot_id_to_species, current_teamsize

        if not player or player not in slot_id_to_species: return None

        # --- Normalize the species we are looking for ---
        normalized_species_lookup = normalize_species_name(species_from_details)
        if not normalized_species_lookup or normalized_species_lookup == 'unknown':
            print(f"Warning: Lookup requested for invalid/unknown species '{species_from_details}' (normalized to '{normalized_species_lookup}')")
            return None

        # 1. Check Nickname Mapping (Nickname itself is NOT normalized here)
        if identifier_name and player in nickname_to_slot_id and identifier_name in nickname_to_slot_id[player]:
            found_slot_id = nickname_to_slot_id[player][identifier_name]
            # Validate: Does the slot still exist?
            if found_slot_id in current_state.get(player, {}):
                # FIX #5: Check CURRENT species first (handles forme changes)
                current_slot_species = current_state[player][found_slot_id].get('species')
                if current_slot_species == normalized_species_lookup:
                    return found_slot_id
                # Also check if stored initial species matches
                initial_species = slot_id_to_species.get(player, {}).get(found_slot_id)
                if initial_species == normalized_species_lookup:
                    return found_slot_id

        # FIX #5: 2a. Check CURRENT state species (handles forme changes)
        for slot_id, poke_state in current_state.get(player, {}).items():
            if isinstance(poke_state, dict):
                current_species = poke_state.get('species')
                if current_species == normalized_species_lookup:
                    # Found match in current state
                    raw_species_name = species_from_details
                    if identifier_name and identifier_name != raw_species_name:
                        if identifier_name not in nickname_to_slot_id.get(player, {}) or nickname_to_slot_id[player].get(identifier_name) == slot_id:
                            nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                    return slot_id

        # 2b. Check Stored Initial Normalized Species Mapping (Exact Match)
        for slot_id, stored_normalized_species in slot_id_to_species.get(player, {}).items():
            if stored_normalized_species == normalized_species_lookup:
                # Found match based on normalized species. Update nickname map if applicable.
                raw_species_name = species_from_details
                if identifier_name and identifier_name != raw_species_name:
                    if identifier_name not in nickname_to_slot_id.get(player, {}) or nickname_to_slot_id[player].get(identifier_name) == slot_id:
                        nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                return slot_id

        # 2c. Fuzzy Match - Try base species variants (e.g., mimikyubusted -> mimikyu)
        lookup_candidates = get_base_species(normalized_species_lookup)
        for candidate in lookup_candidates[1:]:  # Skip first (already tried exact match)
            for slot_id, stored_normalized_species in slot_id_to_species.get(player, {}).items():
                # Check if stored species matches any candidate
                if stored_normalized_species == candidate:
                    if identifier_name:
                        nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                    return slot_id
                # Also check if stored species could be a forme of the same base
                stored_candidates = get_base_species(stored_normalized_species)
                if candidate in stored_candidates or normalized_species_lookup in stored_candidates:
                    if identifier_name:
                        nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                    return slot_id

        # 3. Dynamic Slot Creation - For replays without |poke| lines
        # FIX #8: Use current_state length instead of slot_id_to_species
        existing_slots = len(current_state.get(player, {}))
        max_slots = current_teamsize.get(player, 6) if current_teamsize.get(player, 0) > 0 else 6
        
        if existing_slots < max_slots:
            new_slot_id = f'slot{existing_slots + 1}'
            # Initialize the slot if it doesn't exist
            if new_slot_id not in current_state.get(player, {}):
                if player not in current_state:
                    current_state[player] = {}
                current_state[player][new_slot_id] = get_initial_pokemon_state()
            
            # Register the species
            slot_id_to_species[player][new_slot_id] = normalized_species_lookup
            current_state[player][new_slot_id]['species'] = normalized_species_lookup
            
            # Map nickname if provided
            if identifier_name:
                nickname_to_slot_id.setdefault(player, {})[identifier_name] = new_slot_id
            
            print(f"INFO: Dynamically created {new_slot_id} for {player} with species '{normalized_species_lookup}'")
            return new_slot_id

        # 4. Error Case - Lookup Failed and no room for new slots
        print(f"ERROR: Could not find slot_id for {player} using nickname='{identifier_name}' OR normalized_species='{normalized_species_lookup}'.")
        print(f"  Initial Normalized Species Map: {slot_id_to_species.get(player)}")
        print(f"  Current State Species: {[(sid, current_state[player][sid].get('species')) for sid in current_state.get(player, {}) if isinstance(current_state[player].get(sid), dict)]}")
        print(f"  Known Nicknames: {nickname_to_slot_id.get(player)}")
        return None

    # --- Phase 2: Process Turns and Events ---
    last_action_player = None
    for line_num, line in enumerate(lines):
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]

        try:
            # --- Basic Turn/Winner/Faint/Tera handling ---
            if event_type == 'turn':
                current_state['turn_number'] = int(parts[2])
                last_action_player = 'p2' if current_state['turn_number'] > 0 else None
                current_state['player_to_move'] = 'p1'
                # FIX #6: Reset action tracking for new turn
                action_recorded_this_turn = {'p1': False, 'p2': False}

            elif event_type == 'win':
                winner_name = parts[2]
                if player_names['p1'] and player_names['p2']:
                    if player_names['p1'] == winner_name: current_state['battle_winner'] = 'p1'
                    elif player_names['p2'] == winner_name: current_state['battle_winner'] = 'p2'
                    else: current_state['battle_winner'] = 'unknown'
                else:
                    current_state['battle_winner'] = 'unknown_players_not_parsed'

            elif event_type == 'faint':
                if len(parts) >= 3:
                    identifier = parts[2]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state:
                                poke_state['hp_perc'] = 0
                                poke_state['status'] = 'fnt'
                                poke_state['is_fainted'] = True
                                poke_state['is_active'] = False
                            active_slot[player] = None

            elif event_type == '-terastallize':
                if len(parts) >= 4:
                    identifier, tera_type = parts[2], parts[3]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state:
                                poke_state['terastallized'] = True
                                poke_state['tera_type'] = tera_type

            # --- Core State Updates (HP, Status, Boosts) ---
            elif event_type in ['-damage', '-heal', '-sethp']:
                if len(parts) >= 4:
                    identifier, hp_str, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state:
                                hp_perc = parse_hp(hp_str)
                                if hp_perc is not None:
                                    poke_state['hp_perc'] = hp_perc
                                    if hp_perc == 0 and poke_state['status'] != 'fnt':
                                        poke_state['status'] = 'fnt'
                                        poke_state['is_fainted'] = True
                                        poke_state['is_active'] = False

            elif event_type == '-status':
                if len(parts) >= 4:
                    identifier, status, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state: poke_state['status'] = status.lower()

            elif event_type == '-curestatus':
                if len(parts) >= 4:
                    identifier, status, *_ = parts[2:]
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        # Try to find the slot - could be active OR benched (e.g., Aromatherapy, Heal Bell)
                        slot_id = active_slot.get(player)
                        # If identifier has details, try to look up the specific Pokemon
                        if ident_name_part:
                            # Check nickname map first
                            if ident_name_part in nickname_to_slot_id.get(player, {}):
                                slot_id = nickname_to_slot_id[player][ident_name_part]
                            else:
                                # Try to match by normalized species name
                                normalized_ident = normalize_species_name(ident_name_part)
                                for sid, species in slot_id_to_species.get(player, {}).items():
                                    if species == normalized_ident:
                                        slot_id = sid
                                        break
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state and poke_state['status'] == status.lower():
                                poke_state['status'] = 'none'

            elif event_type == '-boost' or event_type == '-unboost':
                if len(parts) >= 5:
                    identifier, stat, amount_str, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state and 'boosts' in poke_state:
                                try:
                                    stat_key = stat.lower()
                                    amount_val = int(amount_str)
                                    if stat_key in poke_state['boosts']:
                                        mult = 1 if event_type == '-boost' else -1
                                        boosts = poke_state['boosts']
                                        boosts[stat_key] = max(-6, min(6, boosts[stat_key] + (mult * amount_val)))
                                except (ValueError, KeyError):
                                    print(f"Warning: Invalid stat/amount '{stat}/{amount_str}' in {event_type}. Line: {line.strip()}")

            elif event_type == '-setboost':
                if len(parts) >= 5:
                    identifier, stat, amount_str, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state and 'boosts' in poke_state:
                                try:
                                    stat_key = stat.lower()
                                    amount_val = int(amount_str)
                                    if stat_key in poke_state['boosts']:
                                        poke_state['boosts'][stat_key] = max(-6, min(6, amount_val))
                                except (ValueError, KeyError): pass

            elif event_type == '-clearboost':
                if len(parts) >= 3:
                    identifier = parts[2]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state and 'boosts' in poke_state:
                                poke_state['boosts'] = {k: 0 for k in poke_state['boosts']}

            elif event_type == '-clearallboost':
                for player in ['p1', 'p2']:
                    for slot_id, poke_state in current_state.get(player, {}).items():
                        if isinstance(poke_state, dict) and 'boosts' in poke_state:
                            poke_state['boosts'] = {k: 0 for k in poke_state['boosts']}

            elif event_type == '-clearnegativeboost':
                if len(parts) >= 3:
                    identifier = parts[2]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state and 'boosts' in poke_state:
                                for stat in poke_state['boosts']:
                                    if poke_state['boosts'][stat] < 0:
                                        poke_state['boosts'][stat] = 0

            # --- Switch / Drag ---
            elif event_type in ['switch', 'drag']:
                if len(parts) >= 5:
                    identifier, details, hp_str, *_ = parts[2:5]
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        # Extract RAW species name first
                        raw_incoming_species = details.split(',')[0].replace('-*', '')
                        # Pass the RAW species name to find_slot_id (it will normalize internally for lookup)
                        incoming_slot_id = find_slot_id(player, ident_name_part, raw_incoming_species)

                        if incoming_slot_id:
                            # Deactivate old Pokémon
                            old_active_slot_id = active_slot.get(player)
                            if old_active_slot_id and player in current_state and old_active_slot_id in current_state[player]:
                                old_poke_state = current_state[player].get(old_active_slot_id)
                                if old_poke_state: old_poke_state['is_active'] = False

                            active_slot[player] = incoming_slot_id
                            hp_perc = parse_hp(hp_str)
                            if hp_perc is None: hp_perc = 100

                            if incoming_slot_id in current_state.get(player, {}):
                                poke_state = current_state[player][incoming_slot_id]
                                # Store the NORMALIZED species name in the state
                                poke_state['species'] = normalize_species_name(raw_incoming_species)
                                poke_state['hp_perc'] = hp_perc
                                poke_state['is_active'] = True
                                poke_state['is_fainted'] = (hp_perc == 0)
                                poke_state['revealed'] = True
                                # Reset boosts on switch-in (standard Pokemon battle mechanics)
                                poke_state['boosts'] = {k: 0 for k in poke_state.get('boosts', get_initial_pokemon_state()['boosts'])}
                                # NOTE: Tera state persists for entire battle, do NOT reset

            # --- Field State Updates ---
            elif event_type == '-weather':
                weather_state = parts[2].split('[from]')[0].split('[upkeep]')[0].strip().lower()
                current_state['field']['weather'] = weather_state if weather_state != 'none' else 'none'

            elif event_type == '-terrain':
                terrain_state = parts[2].split('[from]')[0].strip().lower()
                current_state['field']['terrain'] = terrain_state if terrain_state != 'none' else 'none'

            elif event_type == '-fieldstart':
                if len(parts) >= 3:
                    condition = parse_condition_name(parts[2])
                    if condition in PSEUDO_WEATHERS:
                        current_state['field']['pseudo_weather'] = condition.lower()

            elif event_type == '-fieldend':
                if len(parts) >= 3:
                    condition = parse_condition_name(parts[2]).lower()
                    if current_state['field']['pseudo_weather'] == condition:
                        current_state['field']['pseudo_weather'] = 'none'

            # --- Side Condition / Hazard Updates ---
            elif event_type == '-sidestart':
                if len(parts) >= 4:
                    side_identifier, condition_str = parts[2], parts[3]
                    affected_side_player = parse_player_from_side(side_identifier)
                    condition = parse_condition_name(condition_str)

                    if affected_side_player and condition:
                        target_hazard_key = f'{affected_side_player}_hazards'
                        target_side_key = f'{affected_side_player}_side'

                        if condition in HAZARD_CONDITIONS and target_hazard_key in current_state['field']:
                            hazard_dict = current_state['field'][target_hazard_key]
                            if condition in hazard_dict:
                                max_layers = 3 if condition == 'Spikes' else (2 if condition == 'Toxic Spikes' else 1)
                                hazard_dict[condition] = min(max_layers, hazard_dict[condition] + 1)
                        elif condition in SIDE_CONDITIONS and condition not in HAZARD_CONDITIONS and target_side_key in current_state['field']:
                            side_dict = current_state['field'][target_side_key]
                            if condition in side_dict: side_dict[condition] = 1

            elif event_type == '-sideend':
                if len(parts) >= 4:
                    side_identifier, condition_str = parts[2], parts[3]
                    affected_side_player = parse_player_from_side(side_identifier)
                    condition = parse_condition_name(condition_str)

                    if affected_side_player and condition:
                        target_hazard_key = f'{affected_side_player}_hazards'
                        if condition in HAZARD_CONDITIONS and target_hazard_key in current_state['field']:
                            hazard_dict = current_state['field'][target_hazard_key]
                            if condition in hazard_dict:
                                hazard_dict[condition] = 0
                        target_side_key = f'{affected_side_player}_side'
                        if condition in SIDE_CONDITIONS and condition not in HAZARD_CONDITIONS and target_side_key in current_state['field']:
                            side_dict = current_state['field'][target_side_key]
                            if condition in side_dict: side_dict[condition] = 0

            # --- Forme Change Handler ---
            elif event_type == '-formechange' or event_type == 'detailschange':
                if len(parts) >= 4:
                    identifier, details, *_ = parts[2:]
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state:
                                # Extract new species from details and normalize
                                raw_new_species = details.split(',')[0].replace('-*', '')
                                normalized_new_species = normalize_species_name(raw_new_species)
                                # FIX #5: Update current species (find_slot_id now checks this)
                                poke_state['species'] = normalized_new_species
                                # Note: We don't update slot_id_to_species because we want
                                # to keep the original species for initial lookup

            # --- Action Recording ---
            # FIX #1: Removed duplicate switch processing here
            # Switch events are already handled above at lines for switch/drag
            
            is_player_action = False
            action_description = None
            acting_player = None
            move_used_this_action = None

            if event_type == 'move':
                if len(parts) >= 4:
                    identifier, move_name_raw, *_ = parts[2:]
                    # FIX #7: Normalize move name for consistency
                    move_name = normalize_move_name(move_name_raw)
                    
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        acting_player = player
                        is_player_action = True
                        action_description = f"move:{move_name}"
                        move_used_this_action = move_name

                        # Update nickname map based on move identifier
                        slot_that_moved = active_slot.get(acting_player)
                        if slot_that_moved and slot_that_moved in slot_id_to_species.get(acting_player, {}):
                            initial_species = slot_id_to_species[acting_player][slot_that_moved]
                            if ident_name_part and normalize_species_name(ident_name_part) != initial_species:
                                current_mapping = nickname_to_slot_id.get(acting_player, {}).get(ident_name_part)
                                if current_mapping is None or current_mapping == slot_that_moved:
                                    nickname_to_slot_id.setdefault(acting_player, {})[ident_name_part] = slot_that_moved

            # FIX #1: Switch actions are recorded differently now
            # We record the switch action at the time of switch event
            elif event_type in ['switch', 'drag']:
                if len(parts) >= 4:
                    identifier, details, *_ = parts[2:]
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    raw_species = details.split(',')[0].replace('-*', '')
                    if player:
                        acting_player = player
                        is_player_action = True
                        # FIX #7: Use normalized species name for action label
                        action_description = f"switch:{normalize_species_name(raw_species)}"

            # --- State Snapshotting and Update Last Move / Revealed Moves ---
            # FIX #6: Only snapshot if action not already recorded this turn (prevents multi-hit duplicates)
            if is_player_action and acting_player and not action_recorded_this_turn.get(acting_player, False):
                # TIMING NOTE: Snapshot captures state BEFORE this action
                # last_move_seen contains what happened in PREVIOUS action
                state_snapshot = copy.deepcopy(current_state)
                state_snapshot['last_move_p1'] = last_move_seen['p1']
                state_snapshot['last_move_p2'] = last_move_seen['p2']
                state_snapshot['action_taken'] = action_description
                state_snapshot['player_to_move'] = acting_player
                game_states.append(state_snapshot)
                
                # FIX #6: Mark that this player has had their action recorded this turn
                action_recorded_this_turn[acting_player] = True

                # Update main state AFTER snapshotting
                if move_used_this_action:
                    # FIX #7: Store normalized move name
                    last_move_seen[acting_player] = move_used_this_action
                    slot_that_moved = active_slot.get(acting_player)
                    if slot_that_moved and acting_player in current_state and slot_that_moved in current_state[acting_player]:
                        poke_state = current_state[acting_player].get(slot_that_moved)
                        if poke_state and isinstance(poke_state.get('revealed_moves'), set):
                            # FIX #7: Add normalized move name
                            poke_state['revealed_moves'].add(move_used_this_action)

                current_state['player_to_move'] = 'p2' if acting_player == 'p1' else 'p1'
                last_action_player = acting_player

        except Exception as e:
            print(f"\n!!! PARSING ERROR on line {line_num+1} for Replay ID {replay_id} !!!")
            print(f"Line content: {line.strip()}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            traceback.print_exc()
            print("--- Continuing parsing ---")

    # --- Finalize ---
    final_winner = current_state.get('battle_winner')
    for state in game_states:
        state['battle_winner'] = final_winner
        state.setdefault('last_move_p1', 'none')
        state.setdefault('last_move_p2', 'none')

    return game_states


# --- Flattening function ---
def flatten_state(state_dict):
    """Converts a nested state dictionary into a flat dictionary for DataFrame creation."""
    flat = {}
    flat['replay_id'] = state_dict.get('replay_id', 'unknown')
    flat['turn_number'] = state_dict.get('turn_number', 0)
    flat['player_to_move'] = state_dict.get('player_to_move')
    flat['action_taken'] = state_dict.get('action_taken')
    flat['battle_winner'] = state_dict.get('battle_winner')
    flat['last_move_p1'] = state_dict.get('last_move_p1', 'none')
    flat['last_move_p2'] = state_dict.get('last_move_p2', 'none')

    # Pokemon States
    for player in ['p1', 'p2']:
        num_slots_to_generate = 6
        for i in range(num_slots_to_generate):
            slot_id = f'slot{i+1}'
            prefix = f'{player}_{slot_id}'
            if player in state_dict and isinstance(state_dict[player], dict) and slot_id in state_dict[player]:
                poke_state = state_dict[player][slot_id]
                if isinstance(poke_state, dict):
                    # Species should already be normalized from parsing
                    flat[f'{prefix}_species'] = poke_state.get('species', normalize_species_name('Unknown'))
                    flat[f'{prefix}_hp_perc'] = poke_state.get('hp_perc', 100)
                    flat[f'{prefix}_status'] = poke_state.get('status', 'none')
                    flat[f'{prefix}_is_active'] = int(poke_state.get('is_active', False))
                    flat[f'{prefix}_is_fainted'] = int(poke_state.get('is_fainted', False))
                    flat[f'{prefix}_terastallized'] = int(poke_state.get('terastallized', False))
                    flat[f'{prefix}_tera_type'] = poke_state.get('tera_type', 'none')
                    boosts = poke_state.get('boosts', {})
                    for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                        flat[f'{prefix}_boost_{stat}'] = boosts.get(stat, 0)
                    moves_set = poke_state.get('revealed_moves', set())
                    if isinstance(moves_set, set):
                        # FIX #7: Moves are already normalized when added
                        flat[f'{prefix}_revealed_moves'] = ",".join(sorted(list(moves_set))) if moves_set else 'none'
                    else:
                        flat[f'{prefix}_revealed_moves'] = 'error_state'
                else:
                    # Fill with default/error values
                    flat[f'{prefix}_species'] = normalize_species_name('ErrorState')
                    flat[f'{prefix}_hp_perc'] = 0
                    flat[f'{prefix}_status'] = 'none'
                    flat[f'{prefix}_is_active'] = 0
                    flat[f'{prefix}_is_fainted'] = 1
                    flat[f'{prefix}_terastallized'] = 0
                    flat[f'{prefix}_tera_type'] = 'none'
                    for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat[f'{prefix}_boost_{stat}'] = 0
                    flat[f'{prefix}_revealed_moves'] = 'error_state'
            else:
                # Add placeholder columns
                flat[f'{prefix}_species'] = normalize_species_name('Absent')
                flat[f'{prefix}_hp_perc'] = 0
                flat[f'{prefix}_status'] = 'none'
                flat[f'{prefix}_is_active'] = 0
                flat[f'{prefix}_is_fainted'] = 1
                flat[f'{prefix}_terastallized'] = 0
                flat[f'{prefix}_tera_type'] = 'none'
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat[f'{prefix}_boost_{stat}'] = 0
                flat[f'{prefix}_revealed_moves'] = 'none'

    # Field States
    initial_field = get_initial_field_state()
    if 'field' in state_dict and isinstance(state_dict['field'], dict):
        field_state = state_dict['field']
        flat['field_weather'] = field_state.get('weather', initial_field['weather'])
        flat['field_terrain'] = field_state.get('terrain', initial_field['terrain'])
        flat['field_pseudo_weather'] = field_state.get('pseudo_weather', initial_field['pseudo_weather'])
        # Hazards
        for player_prefix, hazard_key in [('p1', 'p1_hazards'), ('p2', 'p2_hazards')]:
            hazard_dict = field_state.get(hazard_key, initial_field[hazard_key])
            for hazard in HAZARD_CONDITIONS:
                flat_key = f'{player_prefix}_hazard_{normalize_species_name(hazard)}'
                flat[flat_key] = hazard_dict.get(hazard, 0)
        # Side Conditions
        relevant_side_conditions = SIDE_CONDITIONS - HAZARD_CONDITIONS
        for player_prefix, side_key in [('p1', 'p1_side'), ('p2', 'p2_side')]:
            side_dict = field_state.get(side_key, initial_field[side_key])
            for condition in relevant_side_conditions:
                flat_key = f'{player_prefix}_side_{normalize_species_name(condition)}'
                flat[flat_key] = side_dict.get(condition, 0)
    else:
        flat['field_weather'] = initial_field['weather']
        flat['field_terrain'] = initial_field['terrain']
        flat['field_pseudo_weather'] = initial_field['pseudo_weather']
        for player_prefix, hazard_key in [('p1', 'p1_hazards'), ('p2', 'p2_hazards')]:
            for hazard in HAZARD_CONDITIONS: flat[f'{player_prefix}_hazard_{normalize_species_name(hazard)}'] = 0
        relevant_side_conditions = SIDE_CONDITIONS - HAZARD_CONDITIONS
        for player_prefix, side_key in [('p1', 'p1_side'), ('p2', 'p2_side')]:
            for condition in relevant_side_conditions: flat[f'{player_prefix}_side_{normalize_species_name(condition)}'] = 0

    return flat


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Pokemon Showdown replay log files into a single data table (Parquet/CSV).")
    parser.add_argument("input_dir", type=str, help="Directory containing the downloaded .log files.")
    parser.add_argument("output_file", type=str, help="Path to save the output data table (e.g., output.parquet or output.csv).")
    parser.add_argument("--format", choices=['parquet', 'csv'], default='parquet', help="Output file format (parquet or csv). Parquet recommended.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Limit the number of files to process (for testing).")

    args = parser.parse_args()

    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))
    if not log_files:
        print(f"Error: No .log files found in directory: {args.input_dir}")
        exit(1)

    if args.limit:
        print(f"Limiting processing to the first {args.limit} files found.")
        log_files = log_files[:args.limit]

    print(f"Found {len(log_files)} .log files to parse in '{args.input_dir}'.")
    print("\n=== FIXES APPLIED IN THIS VERSION ===")
    print("1. Removed duplicate switch processing")
    print("2. Added HP percentage validation (0-100)")
    print("3. Normalized move names for consistency")
    print("4. Fixed forme change slot lookup")
    print("5. Fixed dynamic slot creation counting")
    print("6. Added multi-hit move deduplication")
    print("7. Improved error logging")
    print("=====================================\n")

    all_flattened_states = []
    parsed_files = 0
    failed_files = 0

    for i, file_path in enumerate(log_files):
        filename = os.path.basename(file_path)
        match = re.search(r'-(\d+)\.log$', filename)
        replay_id = match.group(1) if match else f"unknown_{filename}"

        print(f"Parsing file {i+1}/{len(log_files)}: {filename} (ID: {replay_id}) ...", end=' ', flush=True)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()

            parsed_replay_states = parse_showdown_replay(log_content, replay_id=replay_id)

            if not parsed_replay_states:
                print("Warning: No action states extracted.")
                continue

            flattened_replay_states = [flatten_state(s) for s in parsed_replay_states]
            all_flattened_states.extend(flattened_replay_states)
            print(f"OK ({len(flattened_replay_states)} states added).")
            parsed_files += 1

        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
            failed_files += 1
        except Exception as e:
            print(f"FAIL. Unexpected Error: {e}")
            failed_files += 1
            import traceback
            traceback.print_exc()

    print(f"\nParsing complete.")
    print(f"Successfully parsed (produced states): {parsed_files} files")
    print(f"Failed or produced no states:        {failed_files + (len(log_files) - parsed_files - failed_files)} files")
    print(f"Total states collected:              {len(all_flattened_states)}")

    if not all_flattened_states:
        print("No states collected, skipping output.")
        exit(0)

    print("\nCreating DataFrame...")
    try:
        df = pd.DataFrame(all_flattened_states)
        print("DataFrame created successfully.")

        print("\nChecking for key columns in DataFrame:")
        print(f"  'last_move_p1' exists: {'last_move_p1' in df.columns}")
        print(f"  'last_move_p2' exists: {'last_move_p2' in df.columns}")
        revealed_cols_exist = any(col.endswith('_revealed_moves') for col in df.columns)
        print(f"  '*_revealed_moves' columns exist: {revealed_cols_exist}")

        print(f"\nSaving DataFrame to '{args.output_file}' (format: {args.format})...")
        if args.format == 'parquet':
            df.to_parquet(args.output_file, index=False, engine='pyarrow')
        elif args.format == 'csv':
            df.to_csv(args.output_file, index=False)

        print("DataFrame saved successfully.")
        print("\nFinal DataFrame Info:")
        pd.set_option('display.max_info_columns', 200)
        pd.set_option('display.max_rows', 200)
        df.info(verbose=True, show_counts=True)
        print("\nDataFrame Head (showing normalized species and moves):")
        print(df[[col for col in df.columns if 'species' in col or col=='replay_id' or col=='action_taken']].head())
        print("\nDataFrame Tail:")
        print(df.tail())
        pd.reset_option('display.max_info_columns')
        pd.reset_option('display.max_rows')

    except Exception as e:
        print(f"\nError creating or saving DataFrame: {e}")
        import traceback
        traceback.print_exc()
        if args.format == 'parquet' and 'df' in locals():
            try:
                alt_path = args.output_file.replace('.parquet', '.emergency.csv')
                print(f"\nAttempting to save as emergency CSV to: {alt_path}")
                df.to_csv(alt_path, index=False)
                print("Emergency CSV saved.")
            except Exception as csv_e:
                print(f"Failed to save emergency CSV: {csv_e}")
        exit(1)