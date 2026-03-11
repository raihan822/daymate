import {useEffect} from "react";
import useFetch from "./useFetch.jsx";
import {BASE_URL} from "../api/baseUrl.js";

export default function useWeather(location){

    const {data: weather, loading: isWeatherLoading, fetchData: fetchWeather}
        = useFetch(
        `${BASE_URL}/weather`,
        {params:{lat: location.lat, lon: location.lon}},
        false
    );

    useEffect(()=>{
        if(location.lat && location.lon){
            fetchWeather({params:{lat: location.lat, lon: location.lon}});
        }
    }, [location.lon, location.lat, fetchWeather]);




    const description = weather?.weather?.[0]?.description ?? "";   //<python> weather_description = weather.get('weather')[0].get('description');
    const temperature = weather?.main?.temp ?? "--";    //<python> weather_temperature = weather.get('main').get('temp');
    return {
        weather,
        isWeatherLoading,
        description,
        temperature
    };
}