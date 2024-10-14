/**
 * @name Check for CWE-125 in mitre-eg-1
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-125-mitre-eg-1
 */

import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis

from ArrayExpr ae

where

//choose expressions that represent index and size
ae.getArrayOffset().toString() = "index"

and not ( exists( GuardCondition gc, Expr e, Expr expr | 
    e.toString() = "index" 
    and expr.toString() = "size" 
    and gc.ensuresLt(e, expr, 0, ae.getBasicBlock(), true) ) 
    and lowerBound(ae.getArrayOffset()) >= 0 )

select ae, "cwe_125 found in"+ae.getFile().toString()