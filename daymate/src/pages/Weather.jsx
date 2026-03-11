import useGeoLocation from "../hook/useGeoLocation.jsx";
import useWeather from "../hook/useWeather.jsx";

import {Container, Card, Row, Col, Button} from "react-bootstrap";


export default function Weather(){
    const {location, locError, isLocLoading, fetchLocation} = useGeoLocation();

    // WEATHER:
    const {weather, isWeatherLoading,description, temperature} = useWeather(location);
    const weather_description = description;
    const weather_temperature = temperature;
    const icon = weather?.weather?.[0]?.icon;
    const iconUrl = icon ? `https://openweathermap.org/img/wn/${icon}@2x.png` : null;

    return (
        <Container className="py-5">

            {/* Page Header */}
            <div className="text-center mb-5">
                <h1 className="fw-bold">Weather Dashboard</h1>
                <p className="text-muted">Real-time weather based on your location</p>
            </div>

            {locError && <p className="text-danger text-center">{locError}</p>}

            {/* Main Weather Card */}
            <Card className="shadow border-0 mb-4">
                <Card.Body className="text-center">

                    {isWeatherLoading ? (
                        <p>Loading weather...</p>
                    ) : (
                        <>
                            <h3 className="text-muted">{weather?.name}, {weather?.sys?.country}</h3>

                            {iconUrl && (
                                <img
                                    src={iconUrl}
                                    alt="weather icon"
                                    style={{width:80}}
                                />
                            )}

                            <h1 className="display-4 fw-bold">
                                {weather_temperature}°C
                            </h1>

                            <p className="text-capitalize text-secondary">
                                {weather_description}
                            </p>
                        </>
                    )}

                    <Button
                        variant="outline-primary"
                        onClick={fetchLocation}
                        disabled={isLocLoading}
                    >
                        {isLocLoading ? "Fetching location..." : "Refresh Location"}
                    </Button>

                </Card.Body>
            </Card>

            {/* Weather Details */}
            <Row className="g-4">

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Feels Like</h6>
                            <h4>{weather?.main?.feels_like ?? "--"} °C</h4>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Humidity</h6>
                            <h4>{weather?.main?.humidity ?? "--"}%</h4>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Pressure</h6>
                            <h4>{weather?.main?.pressure ?? "--"} hPa</h4>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Wind Speed</h6>
                            <h4>{weather?.wind?.speed ?? "--"} m/s</h4>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Visibility</h6>
                            <h4>{weather?.visibility ?? "--"} m</h4>
                        </Card.Body>
                    </Card>
                </Col>

                <Col md={4}>
                    <Card className="shadow-sm border-0 text-center">
                        <Card.Body>
                            <h6 className="text-muted">Cloud Cover</h6>
                            <h4>{weather?.clouds?.all ?? "--"}%</h4>
                        </Card.Body>
                    </Card>
                </Col>

            </Row>

        </Container>
    );

}