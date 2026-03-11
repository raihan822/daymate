/*V.V.I--->
** React e return e custom html tag as component render korte hobe Capital case e function gula rakhte hobe, otherwise react thinks it as normal HTML tag. but custom tag gular nam e kono html tag nei jar karone kisui render hobe na
    * Capitalization: Renamed briefIntroduction to BriefIntroduction. This tells React it is a custom component and not a built-in HTML tag.
** react e prottekta sibling list e key/id thaka important.
    * Missing Keys: Added key={i.sl} and key={index}. React requires a unique key prop for every element in a list to track changes and optimize rendering.
** react e <br /> is important to add /> not just <br >
** react e CSS attribute gular naam jemon `border-radius` likhte hobe `borderRadius` (camelCase e)
* */
//HomePage.jsx
// my customHooks:
import useLimitedTimeAlertMsg from "../hook/useLimitedTimeAlertMsg.jsx";

import {useNavigate} from "react-router-dom";
import { Container, Button, Card, Alert } from "react-bootstrap";


// My files:
import {TAB1_NAME, TAB2_NAME} from "../App.jsx";

function BriefIntroduction() {
    return(
        <Card className="shadow-sm border-0 mb-4">
            <Card.Body>
                <h2 className="fw-bold mb-3">
                    Welcome to <span className="text-primary">DayMate</span>
                </h2>

                <p className="text-muted">
                    <strong>DayMate</strong> is an AI-powered assistant that helps you
                    plan your day intelligently by combining
                    <strong> real-time weather data</strong>,
                    <strong> local news</strong>, and
                    <strong> smart recommendations</strong>.
                </p>

                <p className="mb-0">
                    <strong>Scenario:</strong> DayMate analyzes your location,
                    weather conditions, and latest news to generate a
                    personalized daily plan.
                </p>
            </Card.Body>
        </Card>
    );
}

function UserManualInformation() {
    return (
        <Card className="shadow-sm border-0 mb-4">
            <Card.Body>

                <h3 className="fw-bold mb-3">User Guide</h3>

                <p className="text-muted">Follow these simple steps:</p>

                <ol className="mb-4">
                    <li>Open the <strong>{TAB1_NAME}</strong> page.</li>
                    <li>Allow location access.</li>
                    <li>
                        View the automatically updated weather and news.
                    </li>
                    <li>
                        Click <strong>Generate Plan</strong> to get an
                        AI-generated schedule for your day.
                    </li>
                </ol>

                <h5 className="fw-semibold">Examples</h5>

                <ul className="text-muted">
                    <li>Rain forecast → Carry an umbrella or reschedule outdoor plans</li>
                    <li>Clear weather → Suggested outdoor activities</li>
                    <li>Emergency or traffic alerts → Adjust your schedule</li>
                </ul>

                <div className="bg-light p-3 rounded mt-3">
                    <strong>Quick Start:</strong> Go to the
                    <strong> "{TAB1_NAME}" </strong>
                    tab, allow location access, and click
                    <strong> Generate Plan</strong>. <span className='text-muted fst-italic'>Or, Just click the Button below!</span>
                </div>

            </Card.Body>
        </Card>
    );
}

export default function HomePage(){
    const navigate = useNavigate();
    const {show, alertMessage} = useLimitedTimeAlertMsg(
        'Info: First API response may take about 30-60s due to Render wakeup time for the backend.',
        20000,   //show for 20s
        1000    // start delay in ms
    )

    return(
        <Container className="py-5">

            {/* Hero Section */}
            <div className="text-center mb-5">
                <h1 className="fw-bold">Plan Your Day Smarter</h1>
                <p className="text-muted fs-5">
                    AI-powered daily planning based on your weather,
                    location and latest news.
                </p>

                {/*Optional: Alert msg during development*/}
                {show && (<Alert variant="warning" className="mb-4">{alertMessage}</Alert>)}
            </div>

            <BriefIntroduction />
            <UserManualInformation />

            {/* CTA */}
            <div className="text-center mt-4">
                <Button
                    variant="primary"
                    size="lg"
                    onClick={()=>navigate('/plan')}
                >
                    Start Planning Your Day
                </Button>
            </div>

        </Container>
    );

}