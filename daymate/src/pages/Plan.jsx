//Plan.jsx
//[Mukhosto] jei function e useCallBack() use kora hoy sheta jodi useEffect diye call kori, tahole sheta useEffect er dependecy[] teo rakhte hobe. along with other dependable variables on the task.

import useGeoLocation from "../hook/useGeoLocation.jsx";
import {useEffect, useState} from "react";
import {Button} from "react-bootstrap";
import useFetch from "../hook/useFetch.jsx";
import {BASE_URL} from "../api/baseUrl.js";
import LoaderComponent from "../components/loader/LoaderComponent.jsx";
import ReactMarkdown from 'react-markdown';

export default function Plan(){
    // My Custom Hook (reactive) usable just by calling!:
    const {location, isLocLoading, locError, fetchLocation} = useGeoLocation();

    // WEATHER:
    const {data:weather, loading: isWeatherLoading, fetchData: fetchWeather} = useFetch(`${BASE_URL}/weather`, {params:{lat: location.lat, lon: location.lon}}, false);
    useEffect(()=>{
            if(location.lat && location.lon){
                fetchWeather({params:{lat: location.lat, lon: location.lon}});
            }
        }, [location.lon, location.lat, fetchWeather]
    );
    const weather_descrition = weather?.weather?.[0].description?? "No description available";  //<python> weather_descrition = weather.get('weather')[0].get('description');
    const weathre_temperature = weather?.main?.temp?? "Temp not found"; //<python> weathre_temperature = weather.get('main').get('temp');

    // NEWS:
    const country_name = 'bd'
    const {data:news, loading: isNewsLoading, fetchData: fetchNews} = useFetch(`${BASE_URL}/news`, {params:{country_name: country_name}}, true);
    //<python> headlines = [a.get("title") for a in news.get("articles", [])[:10]]  # Safe extraction of the dict.get() value with default value []
    // const headlines = news?.articles?.slice(0, 10).map(a => a.title) || [];

    // PLAN:
    const [generatePlanRequested, setGeneratePlanRequested] = useState(false);
    const [hasGenerated, setHasGenerated] = useState(false);
    // const {data:plan, loading: isPlanLoading, fetchData:fetchPlan} = useFetch(`${BASE_URL}/plan`, {method: "post", data: {lat: location.lat, lon: location.lon, location_name: "bd"}, false);
    const { data: plan, loading: isPlanLoading, fetchData: fetchPlan } = useFetch(`${BASE_URL}/plan`, {method: "post"}, false);


    const handleGeneratePlan = ()=>{
        setGeneratePlanRequested(true);
        // setHasGenerated(true);
        // fetchPlan({
        //     data: {
        //         lat: location.lat,
        //         lon: location.lon,
        //         location_name: "bd"
        //     }
        // });
    }
    useEffect(() => {
        /* User clicks Generate
                    ↓
            generateRequested = true
                    ↓
            Wait until location available
                    ↓
            fetchPlan()
        * */
        if (generatePlanRequested && location.lat && location.lon){
            fetchPlan(
                {
                    data:{
                        lat: location.lat,
                        lon: location.lon,
                        location_name: country_name? country_name : 'us'
                    }
                }
            );
            setHasGenerated(true);
            setGeneratePlanRequested(false);
        }
    }, [generatePlanRequested, location.lat, location.lon, fetchPlan]);

    return (
        <div className={'container'}>
            <h1>DayMate Planner</h1>
            <h2>Generate your Plan for the Day on DayMate</h2>
            {locError && <p className="text-danger">{locError}</p>}


            <div className="row">
                <div className="col-6">
                    <div>Weather: {isWeatherLoading ? "Loading..." : weathre_temperature}</div>
                    <div>Description: {isWeatherLoading ? "Loading..." : weather_descrition}</div>
                </div>
                <div className="col-6">
                    <div>Location: lat={location.lat ?? "N/A"}, lon={location.lon ?? "N/A"}</div>
                    <Button variant="primary" onClick={fetchLocation}>
                        {isLocLoading ? "Fetching..." : "Fetch Location Again"}
                    </Button>
                </div>
            </div>

            <div className="row mt-3">
                <h4>News ({isNewsLoading ? "Loading..." : news?.totalArticles?? 0})</h4>

                {isNewsLoading? <LoaderComponent />:
                    <ul>
                        {news?.articles?.slice(0, 10).map((item, idx) => (
                            <li key={idx}>
                                <a id={item.id} href={item.url} target="_blank" rel="noopener noreferrer">
                                    {item.title}
                                </a>
                            </li>
                        ))}
                    </ul>
                }
            </div>


            <Button className="mt-3" onClick={handleGeneratePlan} disabled={!location.lat || !location.lon || isPlanLoading}>
                {
                    !location.lat
                        ? "Waiting for location..."
                        : isPlanLoading
                            ? "Generating Plan..."
                            : "Generate Plan"
                }
            </Button>
            {hasGenerated && (
                <div className="row mt-3">
                    {isPlanLoading ? <LoaderComponent /> : (
                        <div className="card shadow-sm border-0 my-4">
                            <h4>Here is your Plan for the Day:</h4>
                            <div className="card-body p-4">
                                <div className="ai-content lh-lg text-secondary">
                                    <ReactMarkdown>{plan?.planning ?? "N/A"}</ReactMarkdown>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}