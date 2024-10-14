/**
 * @name Check for CWE-787 in mitre-eg-5
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-787-mitre-eg-5 
 */

import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis
import semmle.code.cpp.controlflow.SSA

from ArrayExpr ae

where 

//get the array expressions that are being written into
exists(AssignExpr assign_expr | assign_expr.getLValue() = ae )

//the array expression is part of a while or for loop
and exists( ForStmt for_stmt, WhileStmt while_stmt |
            ae.getEnclosingElement*() = for_stmt
            or ae.getEnclosingElement*()= while_stmt
    )

//there is no guard condition that ensures the index is greater 
//than or equal to zero
//and not lowerBound( ae.getArrayOffset() ) >=0 

//there is a guard condition that has the index of array expression
and ( exists (GuardCondition gc|
            ae.getArrayOffset().toString().matches( gc.getAChild().toString() )  
            //and gc.getEnclosingElement() = rel_op.getEnclosingElement()
        )

        //there is a relation operation in the while or for statements 
        //that contains the index of the array expressions
        and not exists( RelationalOperation ro, WhileStmt ws, ForStmt fs|
                        (ro.getEnclosingStmt*() = ws 
                                and ( ae.getArrayOffset().toString().matches("%"+ro.getLesserOperand().toString()+"%" ) 
                                        or
                                        ae.getArrayOffset().toString().matches("%"+ro.getGreaterOperand().toString()+"%" ) 
                                        )
                        )
                        or
                        (ro.getEnclosingStmt*() = fs 
                                and ( ae.getArrayOffset().toString().matches("%"+ro.getLesserOperand().toString()+"%" ) 
                                        or
                                        ae.getArrayOffset().toString().matches("%"+ro.getGreaterOperand().toString()+"%" ) 
                                        )
                        )
                )
        or
        lowerBound( ae.getArrayOffset() ) < 0               

)
select  ae, "cwe-787 found in "+ ae.getFile().getShortName()+ " at line" + ae.getLocation().getEndLine()