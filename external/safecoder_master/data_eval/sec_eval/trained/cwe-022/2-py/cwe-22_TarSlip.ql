/**
 * @name Check for CWE-22 in codeql-eg-TarSlip
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-22-codeql-eg-tarslip
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

from DataFlow::CfgNode src, CallNode call, 
    DataFlow::CfgNode sink

where
//The source of open tarfile objects. That is, any call to `tarfile.open(...)'
src.asCfgNode() = Value::named("tarfile.open").getACall() 
and

//sink is the Argument to extract method
(sink.asCfgNode() = call.getFunction().(AttrNode).getObject("extractall")
or
sink.asCfgNode() = call.getFunction().(AttrNode).getObject("extract") )

//check if teh source taints the sink i.e. an open tarfile object
//is called by an extract method
and 
TaintTracking::localTaint( src , sink )

//there is no check on the call of tar extraction
and 
not exists (If if_stmt| 
             if_stmt.getASubStatement().contains( call.getNode() )
            )

select call, "cwe-22: "+call.toString()+ " at line "+ call.getLocation().getStartLine()+
            " is a tar extraction call that is unchecked"
