import { useState, useEffect } from "react";

/**
 * Custom hook to show a message for a limited time
 * @param {string} message - The message to show in the alert
 * @param {number} duration - How long to show the alert (ms). Default 60000ms (1 min)
 * @param {number} startDelay - Optional delay before showing the alert (ms). Default 0 (immediately)
 * @returns {object} { show, setShow, alertMessage } - `show` boolean, `setShow` setter, The alert msg!
 */
export default function useLimitedTimeAlertMsg(message, duration = 60000, startDelay = 0) {
    const [show, setShow] = useState(false);    //start hidden
    const [alertMessage] = useState(message);

    useEffect(() => {
        // Start the alert after startDelay
        const startTimer = setTimeout(() => {
            setShow(true);

            // Hide the alert automatically after duration
            const hideTimer = setTimeout(() => {
                setShow(false);
            }, duration);

            // Cleanup hide timer if component unmounts early
            return () => clearTimeout(hideTimer);

        }, startDelay);

        // Cleanup start timer if component unmounts early
        return () => clearTimeout(startTimer);
    }, [duration, startDelay]);

    return { show, setShow, alertMessage };
}



/*How to use on other component:---------->>
import useLimitedTimeAlertMsg from "../hook/useLimitedTimeAlertMsg.jsx";
//in function:
    const {show, alertMessage} = useLimitedTimeAlertMsg(
            'Info: First API response may take about 30-60s due to Render wakeup time for the backend.',
            30000,   //show for 30s
            2000    // start delay in ms
        )
//in return jsx:    [Optional: Alert msg during development]
    {show && (<Alert variant="warning" className="mb-4">{alertMessage}</Alert>)}
* */