// This Custom Component makes a Table out of a given Object Array [{KEY:VALUE},{KEY:VALUE},{KEY:VALUE}]. Nothing else is required!
import "./TableComponent.css"
export default function TableComponent({
                                         caption="",
                                         dataObjArray = [],   //[This is the --MAIN-- Data from API] data api table e majhe majhe na o thakte pare
                                         excludedKeys = [],   //not important for elsewhere. just optional for what to show.
                                         footNote="" }){
    //Default case: if no data on the array:
    if (dataObjArray.length === 0) return <p>No data available</p>;

    //Otherwise: Continue:
    const KEY_NAMES = Object.keys(dataObjArray[0]); //list of obj, so took the fist obj only to see its key names.  //returns a list of strings
    const ALLOWED_KEY_NAMES = KEY_NAMES.filter(i => !excludedKeys.includes(i))
    const NUMBER_OF_COLUMNS = ALLOWED_KEY_NAMES.length;
    const SL = KEY_NAMES[0];    //first key should be sent as serial number of the data

    /*  Note:-
    usee this for CAPITALISED WORD:=>   text.toUpperCase()
    use for first Char Capped in JS:=>    text.charAt(0).toUpperCase() + text.slice(1)
    * */
    return (
        <table>
            {caption !== "" && <caption><strong>{caption.toUpperCase()}</strong></caption>}
            <thead>
                <tr>
                    {ALLOWED_KEY_NAMES.map(
                        (key_name, index) =>
                            <th key={index} scope="col">{key_name.charAt(0).toUpperCase() + key_name.slice(1)}</th>
                    )}
                </tr>
            </thead>
            <tbody>
                {dataObjArray.map(
                    (data, indexNo) =>
                        <tr key={data[SL] || indexNo}>
                            {ALLOWED_KEY_NAMES.map( (key_name, idx) =>
                                    <td key={idx} scope="col">{data[key_name]}</td>
                            )}
                        </tr>
                )}
            </tbody>
            <tfoot>
                {footNote!=="" &&
                    <tr>
                        <td colSpan={NUMBER_OF_COLUMNS}>{footNote}</td>
                    </tr>}
            </tfoot>
        </table>
    )
}