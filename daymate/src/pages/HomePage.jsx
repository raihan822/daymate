/*V.V.I--->
** React e return e custom html tag as component render korte hobe Capital case e function gula rakhte hobe, otherwise react thinks it as normal HTML tag. but custom tag gular nam e kono html tag nei jar karone kisui render hobe na
    * Capitalization: Renamed briefIntroduction to BriefIntroduction. This tells React it is a custom component and not a built-in HTML tag.
** react e prottekta sibling list e key/id thaka important.
    * Missing Keys: Added key={i.sl} and key={index}. React requires a unique key prop for every element in a list to track changes and optimize rendering.
** react e <br /> is important to add /> not just <br >
** react e CSS attribute gular naam jemon `border-radius` likhte hobe `borderRadius` (camelCase e)
* */
//HomePage.jsx
import {useNavigate} from "react-router-dom";
import { Container, Button } from "react-bootstrap";


// My files:
import {TAB1_NAME, TAB2_NAME} from "../App.jsx";

function BriefIntroduction() {
    return(
        <div>
            <h2>Welcome to <strong>DayMate</strong></h2>
            <p>
                <strong>DayMate</strong> - an AI-powered assistant that helps users plan their day by combining
                real-time weather data, local news, and intelligent recommendations.

                <br /><strong>Scenario:</strong> DayMate analyzes current weather and local news to provide personalized daily planning
                suggestions.
            </p>

        </div>
    );
}

function UserManualInformation() {
    return (
        <div>
            <h2>USER MANUAL</h2>
            <p>How to use:</p>
            <ol>
                <li>Go to Plan to Make a Plan for the day</li>
                <li>Give permission to Location access and see the Weather and News getting updated</li>
            </ol>
            <strong>Examples:</strong>
            <ul>
                <li>Rain forecasted → Suggest carrying an umbrella or rescheduling outdoor plans</li>
                <li>Clear weather → Recommend outdoor activities</li>
                <li>Traffic alerts or emergencies → Advise schedule modifications</li>
            </ul>
             <pre>
                 <strong>Quick Starter:</strong> Go to the <strong>"{TAB1_NAME}"</strong>, give permission for location access and Generate a plan for the day.
             </pre>
         </div>
     );
}

export default function HomePage(){
    const navigate = useNavigate();
    return(
        <Container>
            <BriefIntroduction />
            <UserManualInformation />

            <div className="d-flex justify-content-center gap-2 mb-2">
                <Button variant="primary" size="lg" onClick={ ()=>{navigate('/plan')} }>Go to Plan</Button>
            </div>
        </Container>
    )

}