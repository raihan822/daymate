//Plan.jsx
import useGeoLocation from "../hook/useGeoLocation.jsx";
import {useEffect, useState} from "react";
import {Button} from "react-bootstrap";

export default function Plan(){
    // My Custom Hook (reactive) usable just by calling!:
    const {location, isLoading, error, fetchLocation} = useGeoLocation();


    return (
        <div className={'container'}>
            <h1>DayMate Planner</h1>
            <h2>Generate your Plan for the Day on DayMate</h2>
            {error && <p className="text-danger">{error}</p>}

            <div className={'row'}>
                <div className={'col-6 justify-content-start'}>
                    <div className={'row'}>
                        Weather: degree=weather_dgr_here,
                    </div>
                    <div className={'row'}>
                        Description: weather_desc_here,
                    </div>
                </div>
                <div className={'col-6 justify-content-end'}>
                    <div className={'row'}>
                        Location: lat={location.lat}, lon={location.lon}
                    </div>
                    <div className={'row'}>
                        <Button variant={'primary'} onClick={()=>{fetchLocation()}}>
                            {isLoading ? 'Fetching...' : 'Fetch Location Again'}
                        </Button>
                    </div>
                </div>
            </div>

            <div className={'row d-flex justify-content-center'}>
                News Will be shown here [quantity 10]..
            </div>
            <div className={'row d-flex justify-content-center'}>
                <Button variant={'primary'}>Generate Plan</Button>
            </div>
            <div className={'row d-flex justify-content-center'}>
                Planning Result will be shown here...
            </div>

        </div>
    );
}