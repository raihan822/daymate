//Plan.jsx
//[Mukhosto] jei function e useCallBack() use kora hoy sheta jodi useEffect diye call kori, tahole sheta useEffect er dependecy[] teo rakhte hobe. along with other dependable variables on the task.

import {BASE_URL} from "../api/baseUrl.js";
import {useEffect, useState} from "react";
import {Alert, Button} from "react-bootstrap";

// My CustomHooks:
import useGeoLocation from "../hook/useGeoLocation.jsx";
import useFetch from "../hook/useFetch.jsx";
import useLimitedTimeAlertMsg from "../hook/useLimitedTimeAlertMsg.jsx";
    //api based customHooks:
import useWeather from "../hook/useWeather.jsx";
import useNews from "../hook/useNews.jsx";

// My components:
import LoaderComponent from "../components/loader/LoaderComponent.jsx";
import ReactMarkdown from 'react-markdown';

export default function Plan(){
    // My Custom Hook (reactive) usable just by calling!:
    const {location, isLocLoading, locError, fetchLocation} = useGeoLocation();
    const {show, alertMessage} = useLimitedTimeAlertMsg(
        'Info: First API response may take about 30-60s due to Render wakeup time for the backend.',
        20000,   //show for 20s
        1000    // start delay in ms
    )

    // WEATHER:
    const {weather, isWeatherLoading, description, temperature} = useWeather(location);

    const weather_descrition = description;
    const weathre_temperature = temperature;
    const _weather_humidity = weather?.main?.humidity ?? "--";  //etc you can get from api

    // NEWS:
    const country_name = 'bd'
    const {news, isNewsLoading, headlines, infoMessage} = useNews(country_name);
    const headlineCount = isNewsLoading
        ? "..."
        : infoMessage
            ? 0
            : news?.totalArticles ?? 0;

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
        <div className="container py-5">
            {/* HEADER */}
            <div className="text-center mb-5">
                <h1 className="fw-bold">DayMate Planner</h1>
                <p className="text-muted fs-5">
                    Smart daily planning based on your weather, location and news
                </p>

                {/*Optional: Alert msg during development*/}
                {show && (<Alert variant="warning" className="mb-4">{alertMessage}</Alert>)}
            </div>

            {locError && <p className="text-danger text-center">{locError}</p>}

            {/* WEATHER + LOCATION */}
            <div className="row g-4 mb-4 justify-content-between">

                {/* WEATHER CARD */}
                <div className="col-md-6">
                    <div className="card shadow-sm border-0 h-100 w-100">
                        <div className="card-body justify-content-center">
                            <h5 className="card-title mb-3">🌤 Weather</h5>

                            {isWeatherLoading ? (<LoaderComponent />) : (
                                <>
                                    <h2 className="fw-bold">{weathre_temperature}°C</h2>
                                    <p className="text-muted text-capitalize">
                                        {weather_descrition}
                                    </p>
                                </>
                            )}
                        </div>
                    </div>
                </div>

                {/* LOCATION CARD */}
                <div className="col-md-6">
                    <div className="card shadow-sm border-0 h-100 w-100">
                        <div className="card-body justify-content-center">
                            <h5 className="card-title mb-3">📍 Location</h5>

                            <p className="text-muted">
                                Lat: <b>{location.lat ?? "N/A"}</b>
                                <br />
                                Lon: <b>{location.lon ?? "N/A"}</b>
                            </p>

                            <Button variant="outline-primary" onClick={fetchLocation}>
                                {isLocLoading ? "Fetching..." : "Refresh Location"}
                            </Button>
                        </div>
                    </div>
                </div>
            </div>

            {/* NEWS SECTION */}
            <div className="card shadow-sm border-0 mb-4">
                <div className="card-body">
                    <h5 className="mb-3">
                        📰 News
                        <span className="badge bg-secondary ms-2">{headlineCount}</span>
                    </h5>

                    {infoMessage && (
                        <Alert variant="warning" className="mb-4">
                            {infoMessage}
                        </Alert>
                    )}

                    {isNewsLoading ? (<LoaderComponent />) : headlines.length === 0 ? (
                        <p className="text-center text-muted">
                            No news articles available right now.
                        </p>
                    ) : (<ul className="list-group list-group-flush">
                            {news?.articles?.slice(0, 10).map((item, idx) => (
                                <li key={idx} className="list-group-item">
                                    <a
                                        href={item.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-decoration-none"
                                    >
                                        {item.title}
                                    </a>
                                </li>
                            ))}
                        </ul>)}
                </div>
            </div>

            {/* GENERATE PLAN BUTTON */}
            <div className="text-center mb-4">
                <Button
                    size="lg"
                    variant="primary"
                    onClick={handleGeneratePlan}
                    disabled={!location.lat || !location.lon || isPlanLoading}
                >
                    {!location.lat
                        ? "Waiting for location..."
                        : isPlanLoading
                            ? "Generating Plan..."
                            : "Generate My Day Plan"}
                </Button>
            </div>

            {/* AI PLAN */}
            {hasGenerated && (
                <div className="card shadow border-0">
                    <div className="card-body p-4">
                        <h4 className="mb-3">🧠 Your AI Generated Plan</h4>
                        {isPlanLoading ? (<LoaderComponent />) : (
                            <div className="ai-content lh-lg text-secondary">
                                <ReactMarkdown>
                                    {plan?.planning ?? "No plan generated."}
                                </ReactMarkdown>
                            </div>
                        )}
                    </div>
                </div>
            )}

        </div>
    );
}