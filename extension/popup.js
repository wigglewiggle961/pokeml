document.addEventListener('DOMContentLoaded', () => {
    const statusDiv = document.getElementById('status');
    const turnInfo = document.getElementById('turn-info');
    const activePokemon = document.getElementById('active-pokemon');
    const opponentPokemon = document.getElementById('opponent-pokemon');
    const tableBody = document.getElementById('predictions-table-body');

    function connect() {
        const socket = new WebSocket('ws://localhost:8765');

        socket.onopen = function() {
            console.log('Connection established with bot server.');
            statusDiv.textContent = 'Connected';
            statusDiv.className = 'status-connected';
        };

        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('Received data:', data);

            // Update the info bar
            turnInfo.textContent = data.turn;
            activePokemon.textContent = data.active_pokemon;
            opponentPokemon.textContent = data.opponent_active_pokemon;

            // Clear the old table data
            tableBody.innerHTML = '';

            // Populate the table with new predictions
            if (data.predictions && data.predictions.length > 0) {
                data.predictions.forEach(pred => {
                    const row = document.createElement('tr');
                    
                    const rankCell = document.createElement('td');
                    rankCell.textContent = pred.rank;
                    row.appendChild(rankCell);

                    const actionCell = document.createElement('td');
                    actionCell.textContent = pred.action_string;
                    row.appendChild(actionCell);

                    const probCell = document.createElement('td');
                    // Format probability to a nice percentage
                    probCell.textContent = `${(pred.probability * 100).toFixed(2)}%`;
                    row.appendChild(probCell);
                    
                    tableBody.appendChild(row);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="3">No predictions available.</td></tr>';
            }
        };

        socket.onclose = function() {
            console.log('Connection closed. Retrying in 5 seconds...');
            statusDiv.textContent = 'Disconnected';
            statusDiv.className = 'status-disconnected';
            tableBody.innerHTML = '<tr><td colspan="3">Connection lost. Trying to reconnect...</td></tr>';
            // Try to reconnect every 5 seconds
            setTimeout(connect, 5000);
        };

        socket.onerror = function(error) {
            console.error('WebSocket Error:', error);
            statusDiv.textContent = 'Error';
            statusDiv.className = 'status-disconnected';
        };
    }

    // Initial connection attempt
    connect();
});