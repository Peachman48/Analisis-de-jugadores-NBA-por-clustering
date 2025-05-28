from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import pandas as pd
import os
import time

# Obtener la lista de todos los jugadores
all_players = players.get_players()

# Tamaño del lote (500 jugadores por archivo)
batch_size = 500

# Ruta de la carpeta de Descargas
downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')

# Crear una lista para almacenar las estadísticas de cada lote
all_stats = []

# Iterar sobre los jugadores en lotes
for i in range(9 * batch_size, len(all_players), batch_size):
    batch = all_players[i:i + batch_size]
    batch_stats = []
    
    for player in batch:
        player_id = player['id']
        player_name = player['full_name']
        
        try:
            # Obtener las estadísticas de la carrera del jugador
            player_stats = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=60)
            df = player_stats.get_data_frames()[0]  # Obtener el DataFrame de estadísticas
            
            # Agregar el nombre del jugador al DataFrame
            df['Player'] = player_name
            
            # Agregar las estadísticas a la lista del lote
            batch_stats.append(df)
            
            print(f"Datos de {player_name} obtenidos correctamente.")
        except Exception as e:
            print(f"Error al obtener datos de {player_name}: {e}")
        
        # Esperar 2 segundos entre solicitudes para evitar sobrecargar la API
        time.sleep(2)
    
    # Combinar las estadísticas del lote en un solo DataFrame
    batch_df = pd.concat(batch_stats, ignore_index=True)
    
    # Guardar el lote en un archivo CSV
    batch_number = (i // batch_size) + 1
    file_path = os.path.join(downloads_path, f'nba_historical_stats_batch_{batch_number}.csv')
    batch_df.to_csv(file_path, index=False)
    
    print(f"Lote {batch_number} guardado en: {file_path}")
    
    # Limpiar la lista del lote para liberar memoria
    batch_stats.clear()

print("Proceso completado.")