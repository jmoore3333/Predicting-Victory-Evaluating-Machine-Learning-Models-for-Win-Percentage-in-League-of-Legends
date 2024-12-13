import requests
import psycopg2
from psycopg2.extras import Json
import time
import json
with open('credentials.json') as f:
    credentials = json.load(f)

api_key = credentials['apiKey']
db_name = credentials['dbname']
db_user = credentials['user']
db_password = credentials['password']
db_host = credentials['host']

def get_summoner_puuid(summoner_id):
    api_url = f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}?api_key={api_key}'
    resp = requests.get(api_url, timeout=10)
    player_info = resp.json()
    
    # Handle cases where 'puuid' is missing
    if 'puuid' not in player_info:
        print(f"PUUID not found for Summoner ID: {summoner_id}")
        return None
    
    return player_info['puuid']


def get_match_ids(puuid):
    api_url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count=1&api_key={api_key}'
    resp = requests.get(api_url, timeout=10)
    
    if resp.status_code == 429:
        print("Rate limit exceeded. Waiting...")
        reset_time = int(resp.headers.get('X-RateLimit-Reset', time.time() + 60))
        time_to_wait = reset_time - time.time()
        time.sleep(max(time_to_wait, 0))
        return get_match_ids(puuid)
    
    match_ids = resp.json()
    
    if not match_ids:
        print(f"No match IDs found for PUUID: {puuid}")
    
    return match_ids[0] if match_ids else None

def fetch_match_data(match_id):
    match_url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}'
    timeline_url = f'https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline?api_key={api_key}'
    
    match_response = requests.get(match_url)
    timeline_response = requests.get(timeline_url)
    
    match_details = match_response.json()
    timeline_data = timeline_response.json()
    
    return match_details, timeline_data

def extract_metrics(match_details, timeline_data):
    if 'info' not in match_details:
        raise ValueError("Missing 'info' key in match_details")

    metrics = []
    reversed_metrics = []
    participant_to_team = {}
    
    for participant in match_details['info']['participants']:
        participant_to_team[participant['participantId']] = participant['teamId']

    # Initialize data structures to track towers, dragon kills, barons, elders, and inhibitors
    towers = {100: 0, 200: 0}
    dragon_kills = {100: 0, 200: 0}
    barons = {100: 0, 200: 0}
    elders = {100: 0, 200: 0}
    grubs = {100: 0, 200: 0}
    rift_herald = {100: 0, 200: 0}
    
    if 'info' not in timeline_data or 'frames' not in timeline_data['info']:
        raise ValueError("Missing 'info' or 'frames' key in timeline_data")
    
    for frame in timeline_data['info']['frames']:
        timestamp = frame['timestamp']
        
        # Initialize metrics for this frame
        team1_gold = 0
        team2_gold = 0
        team1_xp = 0
        team2_xp = 0
        team1_players_alive = 0
        team2_players_alive = 0
        total_gold = 0
        player_gold = {}
        
        for participant_id, participant_data in frame['participantFrames'].items():
            team_id = participant_to_team[int(participant_id)]
            total_gold += participant_data['totalGold']
            player_gold[participant_id] = participant_data['totalGold']

            if team_id == 100:
                team1_gold += participant_data['totalGold']
                team1_xp += participant_data['xp']
                if participant_data['championStats']['health'] > 0:
                    team1_players_alive += 1
            else:
                team2_gold += participant_data['totalGold']
                team2_xp += participant_data['xp']
                if participant_data['championStats']['health'] > 0:
                    team2_players_alive += 1

        # Check to avoid division by zero
        if total_gold > 0:
            player_gold_percentage = {pid: (gold / total_gold) * 100 for pid, gold in player_gold.items()}
        else:
            player_gold_percentage = {pid: 0 for pid in player_gold.keys()}
        team1_dragon_soul = False
        team2_dragon_soul = False
        # Loop through events to track inhibitor destruction and other metrics
        for event in frame['events']:
            if event['type'] == 'BUILDING_KILL':
                if 'buildingType' in event and event['buildingType'] == 'TOWER_BUILDING':
                    if 'teamId' in event:
                        if event['teamId'] == 100:
                            towers[100] += 1
                        else:
                            towers[200] += 1

            elif event['type'] == 'ELITE_MONSTER_KILL':
                if event['monsterType'] == 'DRAGON':
                    if 'killerTeamId' in event:
                        if event['killerTeamId'] == 100:
                            dragon_kills[100] += 1
                        else:
                            dragon_kills[200] += 1
                elif event['monsterType'] == 'BARON_NASHOR':
                    if 'killerTeamId' in event:
                        if event['killerTeamId'] == 100:
                            barons[100] += 1
                        else:
                            barons[200] += 1
                elif event['monsterType'] == 'HORDE':
                    if 'killerTeamId' in event:
                        if event['killerTeamId'] == 100:
                            grubs[100] += 1
                        else:
                            grubs[200] += 1
                elif event['monsterType'] == 'RIFTHERALD':
                    if 'killerTeamId' in event:
                        if event['killerTeamId'] == 100:
                            rift_herald[100] += 1
                        else:
                            rift_herald[200] += 1

        metrics.append({
            'timestamp': timestamp,
            'team1_gold': team1_gold,
            'team2_gold': team2_gold,
            'gold_difference': team1_gold - team2_gold,
            'team1_xp': team1_xp,
            'team2_xp': team2_xp,
            'xp_difference': team1_xp - team2_xp,
            'team1_players_alive': team1_players_alive,
            'team2_players_alive': team2_players_alive,
            'players_alive': team1_players_alive - team2_players_alive,
            'team1_tower_kills': towers[100],
            'team2_tower_kills': towers[200],
            'tower_kill_difference': towers[100] - towers[200],
            'team1_dragon_kills': dragon_kills[100],
            'team2_dragon_kills': dragon_kills[200],
            'dragon_kill_difference': dragon_kills[100] - dragon_kills[200],
            'team1_barons': barons[100],
            'team2_barons': barons[200],
            'team1_elders': elders[100],
            'team2_elders': elders[200],
            'team1_grubs': grubs[100],
            'team2_grubs': grubs[200],
            'team1_rift_herald': rift_herald[100],
            'team2_rift_herald': rift_herald[200],
        })
        reversed_metrics.append({
            'timestamp': timestamp,
            'team1_gold': team2_gold,
            'team2_gold': team1_gold,
            'gold_difference': team2_gold - team1_gold,
            'team1_xp': team2_xp,
            'team2_xp': team1_xp,
            'xp_difference': team2_xp - team1_xp,
            'team1_players_alive': team2_players_alive,
            'team2_players_alive': team1_players_alive,
            'players_alive': team2_players_alive - team1_players_alive,
            'team1_tower_kills': towers[200],
            'team2_tower_kills': towers[100],
            'tower_kill_difference': towers[200] - towers[100],
            'team1_dragon_kills': dragon_kills[200],
            'team2_dragon_kills': dragon_kills[100],
            'dragon_kill_difference': dragon_kills[200] - dragon_kills[100],
            'team1_barons': barons[200],
            'team2_barons': barons[100],
            'team1_elders': elders[200],
            'team2_elders': elders[100],
            'team1_grubs': grubs[200],
            'team2_grubs': grubs[100],
            'team1_rift_herald': rift_herald[200],
            'team2_rift_herald': rift_herald[100],
        })

    return metrics, reversed_metrics, match_details['info']['teams']

def save_to_database(data):
    try:
        connection = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host
        )
        cursor = connection.cursor()

        # Insert data into the database
        insert_query = """
        INSERT INTO match_results (rank, winning_team, metrics)
        VALUES (%s, %s, %s)
        """

        for rank, winning_team, metrics in data:
            cursor.execute(insert_query, (rank, winning_team, Json(metrics)))

        connection.commit()
    except Exception as e:
        print(f"Error saving data to database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Main part of your script
# Fetch data from the API
def add_to_database(rank, number_to_add):
    api_url = f'https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{rank}/I?page=1&api_key={api_key}'
    resp = requests.get(api_url, timeout=10)
    queue_info = resp.json()

    # Extract summoner IDs
    summoner_ids = [entry['summonerId'] for entry in queue_info]

    data = []
    i = 0
    # Loop through each summoner_id
    for summoner_id in summoner_ids:
        puuid = get_summoner_puuid(summoner_id)
        if i == number_to_add:
            break
        if puuid:
            match_id = get_match_ids(puuid)  # Only fetch one match ID

            # Ensure match_id is not None
            if match_id:
                match_details, timeline_data = fetch_match_data(match_id)

                # Check if data is valid before extracting metrics
                if match_details.get('status') and match_details['status'].get('status_code') == 404:
                    print(f"Match details not found for Match ID: {match_id}")
                    continue

                if timeline_data.get('status') and timeline_data['status'].get('status_code') == 404:
                    print(f"Timeline data not found for Match ID: {match_id}")
                    continue

                try:
                    metrics, reversed_metrics, teams = extract_metrics(match_details, timeline_data)
                    if metrics and teams:
                        winning_team = 100 if teams[0]['win'] else 200
                        reversed_winning_team = 200 if winning_team == 100 else 100
                        data.append((rank, winning_team, metrics))
                        data.append((rank, reversed_winning_team, reversed_metrics))
                except ValueError as e:
                    print(f"Error extracting metrics: {e}")
                time.sleep(1)
            else:
                print(f"No match ID found for PUUID: {puuid}")
        time.sleep(2)
        print(i + 1)
        i += 1

    # Save the collected data to the database
    save_to_database(data)

add_to_database('PLATINUM', 200)  