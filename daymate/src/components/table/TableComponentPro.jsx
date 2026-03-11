// This Custom Component makes a Table out of a given Object Array [{KEY:VALUE},{...},{...}], Column Configuration
// USED THIS Component ON THIS PROJECT's: `See All Products` table section!
import "./TableComponentPro.css"
import {getPartialMatch} from "../../utils/utilities.js";  //made by me,oi jsx ta theke onek gula helper function return korte pari ami future e tai export func akare pathaisi (not export default) So, received as {..}

/*  Instructions:
Columns:[] =[
    {header: "column name you want to set", key: "api_key_toMatch", render: ()=> <>single div output</> ;},
    {header: "column name you want to set", key: "api_key_toMatch", render: ()=> <>single div output</> ;},
    ...
]
*/
/** Component Detailed Information:---
 * @param {string} caption - Caption of the Table (Optional)
 * @param {Object[]} dataObjArray - The raw data to display. (Required)
 * @param {Object[]} columns - Configuration for table columns.
 * @param {string} columns[].header - Column Title.
 * @param {string} [columns[].key] - Data API key (required if no render function).
 * @param {Function} [columns[].render] - Custom cell renderer: (rowData) => JSX.
 * @param {string} footNote - Footnote for the Table (Optional)
 */
export default function TableComponentPro({ caption="",
                                            dataObjArray = [],  //[This is --MAIN-- API Data received from API response] data api table e majhe majhe na o thakte pare
                                            columns = [],       //[This is the Column structure] allowed column names, their keys with api, and any render html if any
                                            footNote=""            //can also send a jsx instead with no quote. <>..your footnotes..</> to execute like html  tag
}){
    /*  Note:-
    usee this for CAPITALISED WORD:=>   text.toUpperCase()
    use fort first Char Capped in JS:=>    text.charAt(0).toUpperCase() + text.slice(1)
    * */

    //Default case: if no data on the array:
    if (dataObjArray.length === 0) return <div className="table-empty">No data available</div>;

    //Otherwise: Continue:
    const KEY_NAMES = Object.keys(dataObjArray[0]); //list of obj, so took the fist obj only to see its key names.  //returns a list of strings
    const SL = getPartialMatch(KEY_NAMES[0]);    //first key should be sent as serial number of the data

    const NUMBER_OF_COLUMNS = columns.length;

    return (
        <div className="table-container">
            <table className="table-pro     table table-hover table-bordered table-sm">

                {caption && (
                    <caption className="table-caption">
                        {caption.toUpperCase()}
                    </caption>
                )}

                <thead>
                <tr>
                    {columns.map((col, colIndex) => (
                        <th key={colIndex} scope="col">
                            {col.header}
                        </th>
                    ))}
                </tr>
                </thead>

                <tbody>
                {dataObjArray.map((data, rowIndex) => (
                    <tr key={data[SL] || rowIndex}>
                        {columns.map((col, colIndex) => (
                            <td key={colIndex}>
                                {col.render ? col.render(data) : data[col.key]}
                            </td>
                        ))}
                    </tr>
                ))}
                </tbody>

                {footNote && (
                    <tfoot>
                    <tr>
                        <td colSpan={NUMBER_OF_COLUMNS}>
                            {footNote}
                        </td>
                    </tr>
                    </tfoot>
                )}

            </table>
        </div>
    );
}