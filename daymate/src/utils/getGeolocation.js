//getGeolocation.js
// Regular JS Function style.
/*GET CURRENT LOCATION FROM BROWSER:
* navigator.geolocation.getCurrentPosition() //This function is provided by all modern browser to access user's location
* then use, setPosition( {lat: .., lon:..} ) to get the coordinates!
* */

/*React's Rules of Hooks:
React uses a strict naming convention to ensure Hooks work correctly (maintaining internal state between renders). ESLint enforces these rules to prevent bugs:
1. Custom Hooks: must start with use (e.g., useGeolocation). This tells React (and ESLint) that the function is allowed to call other Hooks like useState and useEffect.
2. React Components: must start with an Uppercase letter (e.g., GetGeolocation). This identifies the function as a UI builder.
3. Regular Functions: (like getGeolocation) are not allowed to contain Hooks because React cannot guarantee they will be called in the same order every time the component renders.
***
* Custom Hook (React)
to make a React based (reactive: when value changes, jsx that using this customHook will re-render auto), just name the file useXXXX.jsx
- You cant use useState, useEffect if you dont name the file useXXX.jsx which makes the file a custom react hook file
Q: When to make a customHook.jsx or a regularFunction.js file?
1. Benefit of Hook (useGeolocation): It is reactive. If the location changes, or when the data finally arrives, any component using that hook will re-render automatically to show the new data. Use this if you want the UI to "wait" or "update" based on the location.
2. Benefit of Regular Function (getGeolocation): It is imperative. You call it only when you need it (e.g., when a user clicks a button). It doesn't force React to re-render anything, making it more efficient for "one-off" tasks.
*/

/*What is Promise?
Promise is like a Code portions returned as like an API response to be used with async-await until result ready!
--> Promise make the portion of the code block like an API result! instead its just a block of code. returns this code portion's result to who calls it and can be used just like API call, with `async func() await` until result is ready from this portion.
* */
export const getGeolocation = () => {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error("Geolocation is not supported by your browser"));
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: position.coords.latitude,
                    lon: position.coords.longitude,
                });
            },
            (error) => reject(error),
            { enableHighAccuracy: true, timeout: 5000 } // Optional config
        );
    });     //promise finish
};          //function finish

/*How to use:
-- Jehetu eta ke promies e morano hoise , so eta k use korte ekhon async-await diye call korte hobe as it virtually returns promise. just like api
-- without promise, the navigator.geolocation.getCurrentPosition() function is just a normal callback function?!
// In your .jsx file (ONCLICK):
const handleClick = async () => {
    const coords = await getGeolocation(); // Code waits here
    console.log(coords);
};
// Or, (Reactive re-renderable style with useEffect()):
const [coords, setCoords] = useState(null);
useEffect(() => {
    const fetchLocation = async () => {
        try {
            const data = await getGeolocation(); // Calls your .js utility
            setCoords(data); // Manually trigger a re-render with the data
        } catch (err) {
            console.error(err);
        }
    };
    fetchLocation();
}, []);     //Run Once on mount
* */





// // LocalStorage variable saving, for later use cases (cache the data):
// // Normal JS function style. (nonreactive)
// export const getGeolocationWithCache = async () => {
//     // 1. Check if we have a cached version in LocalStorage
//     const cached = localStorage.getItem("user_location");
//     if (cached) {
//         const { lat, lon, timestamp } = JSON.parse(cached);
//         // Optional: Check if the cache is older than 30 minutes
//         const isFresh = Date.now() - timestamp < 30 * 60 * 1000;
//         if (isFresh) return { lat, lon, fromCache: true };
//     }
//
//     // 2. If no cache or stale, ask the browser
//     return new Promise((resolve, reject) => {
//         navigator.geolocation.getCurrentPosition(
//             (pos) => {
//                 const newCoords = {
//                     lat: pos.coords.latitude,
//                     lon: pos.coords.longitude,
//                     timestamp: Date.now()
//                 };
//                 // 3. Save to LocalStorage for next time
//                 localStorage.setItem("user_location", JSON.stringify(newCoords));
//                 resolve(newCoords);
//             },
//             (err) => reject(err)
//         );
//     });
// };

/*Note on Caching var in react:
* Global State (.jsx): If you use Redux/Zustand, you would call this utility in your Root component (like App.jsx) and dispatch the result to your store. This way, the location is fetched once when the app starts and is available everywhere.
* */