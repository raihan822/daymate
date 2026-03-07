//useGeoLocation.jsx
//CUSTOM HOOK useGeoLocation() (customHooks are Reactive and live re-renderable):
import {useState, useEffect} from "react";

export default function useGeoLocation(){
    const [location, setLocation] = useState({
        'lat' : null,
        'lon' : null
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchLocation = ()=> {    //retry function
            if (!navigator.geolocation) {
                setError("Geolocation is not supported by your browser");
                return;
            }
            setIsLoading(true);
            setError(null); // Reset error on new attempt

            //Main: getting current location from Browser
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    setLocation({
                        lat: pos.coords.latitude,
                        lon: pos.coords.longitude,
                    });
                    setIsLoading(false); // Stop loading ONLY on success
                },
                (err) => {
                    setIsLoading(false); // Stop loading ONLY on error
                    if (err.code === 1) {
                        setError("Permission Denied. Please enable location in browser settings.");

                        // alerting the user:
                        alert("Location access is blocked. Please click the 'Lock' icon in your browser's address bar to reset permission.");
                        console.error("Location access is blocked. User denied Geolocation access.");
                    } else {
                        setError(err.message);

                        // alerting the user:
                        console.error("Other location error:", err.message);
                    }
                },
                { enableHighAccuracy: true, timeout: 5000 } // Optional config
            );
        }; //finish



    useEffect(() => {
        fetchLocation();
    }, []);

    return {location, error, isLoading, fetchLocation} //js destructuring er moto kore send korlam, keys must match the exact var name.
}

/*How to use Hook way:
// useGeolocation.jsx
const { position } = useGeolocation();
// In your .jsx return:
return <p>Your location is {position.latitude}</p>; // Updates automatically when ready
* */
