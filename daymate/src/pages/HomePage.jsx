/*V.V.I--->
** React e return e custom html tag as component render korte hobe Capital case e function gula rakhte hobe, otherwise react thinks it as normal HTML tag. but custom tag gular nam e kono html tag nei jar karone kisui render hobe na
    * Capitalization: Renamed briefIntroduction to BriefIntroduction. This tells React it is a custom component and not a built-in HTML tag.
** react e prottekta sibling list e key/id thaka important.
    * Missing Keys: Added key={i.sl} and key={index}. React requires a unique key prop for every element in a list to track changes and optimize rendering.
** react e <br /> is important to add /> not just <br >
** react e CSS attribute gular naam jemon `border-radius` likhte hobe `borderRadius` (camelCase e)
* */
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

                <br />
                <br /><strong>Scenario:</strong>
                <br />DayMate analyzes current weather and local news to provide personalized daily planning
                suggestions.

                (<a href="https://documenter.getpostman.com/view/39406886/2sAY4vh3UX"
                    target="_blank"
                    rel="noopener noreferrer">Link Class2</a>) or,
                (<a href={'src/assets/1.1. API-Doc (CRUD Practice API Documentation PDF) (Sobuj).pdf'}
                    target="_blank"
                    rel="noopener noreferrer">PDF</a>)
            </p>

            <br /><strong>Examples:</strong>
            <ul>
                <li>Rain forecasted → Suggest carrying an umbrella or rescheduling outdoor plans</li>
                <li>Clear weather → Recommend outdoor activities</li>
                <li>Traffic alerts or emergencies → Advise schedule modifications</li>
            </ul>
        </div>
    );
}

// function UserManualInformation() {
//     return (
//         <div>
//             <h2>USER MANUAL</h2>
//             <p>This Project Implements CRUD Operations, with the below Sequence:</p>
//             <ol>
//                 <li>(R)EAD PRODUCT List <i>[GET method]</i></li>
//                 <li>(D)ELETE PRODUCT -with ID <i>[GET method]</i></li>
//                 <li>(C)REATE a PRODUCT -with INFO <i>[POST method]</i></li>
//                 <li>(U)PDATE a PRODUCT -with ID <i>[POST method]</i></li>
//             </ol>
//             <pre>
//                 <strong>Quick Starter:</strong> Go to the <strong>"{TAB1_NAME}"</strong>, to see all the products list and can delete/edit etc to start from there.
//             </pre>
//         </div>
//     );
// }

export default function HomePage(){
    return(
        <Container>
            <BriefIntroduction />
            <div className="d-flex justify-content-center gap-2 mb-2">
                <Button variant="primary" size="lg">
                    Go to Plan
                </Button>
            </div>
        </Container>
    )

}