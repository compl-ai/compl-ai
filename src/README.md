## Code structure

- ```benchmarks/```: Code of benchmark implementations and base benchmark class.
- ```components/```: This is a collection of code needed for the benchmarks, but are more general and serve as a toolkit. 
- ```configs/```: This folder contains the base classes for parsing the configurations, built upon pydantic.
- ```contexts/```: Here, connector classes are implemented which facilitate the interaction between the benchmarks and other code components. With the idea in mind, that the runner code doesn't need to be bothered with too many specifcs.
- ```data/```: Containing some base classes and default implementations for retrieving data for the benchmarks. Specifc benchmark implementations can either build on these or just use the abstract interface.
- ```metics/```: Same as in ```data```, but for metrics instead. 
- ```models/```: Contains the base interface and various implementations for the llm models. Also includes a generic proxy implementation which comes between the model class and the main executor, helping with logging various information. In addition to a dummy model class for testing implementations, model classes for interacting with public APIs are also present.
- ```modifiers/```: This is currently only used for robustness benchmarks and provides a general method of running one benchmark on differnet permutated versions of the base dataset implementation provided. May need some work to be used for benchmarks other than the robustness ones.
- ```prompts```: Contains code for formatting different prompts sucht that the benchmarks ideally don't need to know about the model specific prompt construction. This is not fully up-to-date and needs some adjustments, and also deletion of dead code.
- ```results/```: Providing code for handling the results. Either by stroing them in files or adding them to a MongoDB database.
- ```utils```: Code which doesn't have a specific place to go to.
- ```registry.pu```: Code for performing the registration of benchmark modules. It is a bit rusty and could use some simplifications. This methods will be use in the main ```config.py``` to declare code components.
- ```runner.py```: The main execution code of the benchmarks, designed using the iterator pattern in mind.
