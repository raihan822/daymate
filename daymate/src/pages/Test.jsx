/*GET CURRENT LOCATION FROM BROWSER:
* navigator.geolocation.getCurrentPosition() //This function is provided by all modern browser to access user's location
* then use, setPosition( {lat: .., lon:..} ) to get the coordinates!
* */

import { useEffect, useState } from 'react';

export default function LocationComponent() {
    // Check support during initialization to avoid a synchronous effect update
    const [error, setError] = useState(!navigator.geolocation ? "Geolocation is not supported" : null);
    const [position, setPosition] = useState(
        {
            latitude: null,
            longitude: null
        }
    );

    useEffect(() => {
        if (!navigator.geolocation || error) return;    // Only proceed if geolocation exists and there is no initial error

        navigator.geolocation.getCurrentPosition( (pos) => {
                setPosition({
                    latitude: pos.coords.latitude,
                    longitude: pos.coords.longitude,
                });
            },
            (err) => setError(err.message)
        );
    }, []); // Empty dependency array ensures this runs once

    return (
        <div>
            {error ? <p>Error: {error}</p> : (
                <p>Latitude: {position.latitude}, Longitude: {position.longitude}</p>
            )}
        </div>
    );
}

