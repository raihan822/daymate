import {BASE_URL} from "../api/baseUrl.js";
import useFetch from "../hook/useFetch.jsx";
import useGeoLocation from "../hook/useGeoLocation.jsx";


export default function Weather(){
    const {location, locError, isLocLoading, fetchLocation} = useGeoLocation();

    // WEATHER:
    const {data:weather, loading: isWeatherLoading, fetchData: fetchWeather} = useFetch(`${BASE_URL}/weather`, {params:{lat: location.lat, lon: location.lon}}, false);

    return (
        <div>
            <h1>This is the Weather page</h1>
            <p>Show location name, weather degree and description here</p>
        </div>
    )

}