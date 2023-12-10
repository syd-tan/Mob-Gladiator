# Minecraft Mob Gladiator #

Minecraft Mob Gladiator uses Malmo (not MalmoEnv) to run the RL training. As such, the instructions to install Malmo must be used. The README instructions from the [Malmo Github](https://github.com/microsoft/malmo) are included below in the section called Malmö with full detailed instructions regarding setting up the environment. Simplified and general instructions to install the environment and run the model are provided below.  

Follow the [Malmo](https://github.com/microsoft/malmo) instructions to install the environment compatible to your Operating System prior to running this code. 

Note: For Windows, manual installation will likely be required as automated setup may be outdated.

## Dependencies ##
* [Java JDK 8](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html)
* [Python 3.7](https://www.python.org/downloads/release/python-370/)

#### [Windows Specific](https://github.com/Microsoft/malmo/blob/master/doc/install_windows_manual.md) #### 
* [7 Zip](https://7-zip.org/)
  * Download 64-bit x64
* [ffmpeg](https://github.com/ottverse/ffmpeg-builds/raw/master/ffmpeg-20210804-65fdc0e589-win64-static.zip) 
  * Malmo instructions for downloading ffmpeg are outdated
  * ffmpeg can be downloaded from the link above and instructions for [manual installation](https://github.com/Microsoft/malmo/blob/master/doc/install_windows_manual.md) can be followed to add it to path
#### [Linux](https://github.com/microsoft/malmo/blob/master/doc/install_linux.md) #### 
#### [Mac](https://github.com/microsoft/malmo/blob/master/doc/install_macosx.md) #### 

## Setup ##
pip can be used to install Malmo locally. The following instructions are simplified but the full instructions can be found [here](https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md).
* ```pip3 install malmo``` 
  * already in requirements.txt for this project
  * installs Malmo in python with native code package
*  In your directory of choice, run the following command: 
  * ```python3 -c "import malmo.minecraftbootstrap; malmo.minecraftbootstrap.launch_minecraft()"```
    * A subdirectory called ```MalmoPlatform``` will be created that can be run on your platform
    * In this directory, there will be a ```Minecraft``` folder 

Alternatively, if pip can't be used, instructions to download a pre-built version can be found [here](https://github.com/microsoft/malmo#:~:text=Alternatively%2C%20a%20pre%2Dbuilt%20version%20of%20Malmo%20can%20be%20installed%20as%20follows%3A). 

### Launching Minecraft ###
Once the Malmo files built for your environment are downloaded along with dependencies, the ```Minecraft client``` can be launched by running ```launchClient``` in Malmo's Minecraft folder. 

After Minecraft has been fully launched, Minecraft Mob Gladiator can be run.

### Running Minecraft Mob Gladiator ###
Take the following steps to run the ```Minecraft Mob Gladiator```. 

1. Clone this repository into your Malmo folder: ```git clone https://github.com/syd-tan/Mob-Gladiator.git```
2. Install the dependencies from the requirements.txt file: ```pip install -r requirements.txt```
3. If you have not already, launch Minecraft by going into the Minecraft folder in Malmo and run ```launchClient```
4. With Minecraft launched, run ```python3 training-DQN.py```
    1. If ```training_model.tar``` is in your ```Mob_Gladiator``` directory, training will resume from the latest saved point
    2. To restart training from the beginning, delete ```training_model.tar```
5. The agent can be seen training against mobs. To stop the program, press ```Ctrl + C``` 
6. To run evaluation/graphing of statistics, use `project.ipynb` to graph up to 4400 episodes of checkpoints. The notebook assumes that you have continuous checkpoint files spaced 100 episodes apart for graphing.


# Malmö #

Project Malmö is a platform for Artificial Intelligence experimentation and research built on top of Minecraft. We aim to inspire a new generation of research into challenging new problems presented by this unique environment.

[![Join the chat at https://gitter.im/Microsoft/malmo](https://badges.gitter.im/Microsoft/malmo.svg)](https://gitter.im/Microsoft/malmo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/Microsoft/malmo.svg?branch=master)](https://travis-ci.org/Microsoft/malmo) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Microsoft/malmo/blob/master/LICENSE.txt)
----
    
## Getting Started ##

### Malmo as a native Python wheel ###

On common Windows, MacOSX and Linux variants it is possible to use ```pip3 install malmo``` to install Malmo as a python with native code package: [Pip install for Malmo](https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md). Once installed, the malmo Python module can be used to download source and examples and start up Minecraft with the Malmo game mod. 

Alternatively, a pre-built version of Malmo can be installed as follows:

1. [Download the latest *pre-built* version, for Windows, Linux or MacOSX.](https://github.com/Microsoft/malmo/releases)   
      NOTE: This is _not_ the same as downloading a zip of the source from Github. _Doing this **will not work** unless you are planning to build the source code yourself (which is a lengthier process). If you get errors along the lines of "`ImportError: No module named MalmoPython`" it will probably be because you have made this mistake._

2. Install the dependencies for your OS: [Windows](doc/install_windows.md), [Linux](doc/install_linux.md), [MacOSX](doc/install_macosx.md).

3. Launch Minecraft with our Mod installed. Instructions below.

4. Launch one of our sample agents, as Python, C#, C++ or Java. Instructions below.

5. Follow the [Tutorial](https://github.com/Microsoft/malmo/blob/master/Malmo/samples/Python_examples/Tutorial.pdf) 

6. Explore the [Documentation](http://microsoft.github.io/malmo/). This is also available in the readme.html in the release zip.

7. Read the [Blog](http://microsoft.github.io/malmo/blog) for more information.

If you want to build from source then see the build instructions for your OS: [Windows](doc/build_windows.md), [Linux](doc/build_linux.md), [MacOSX](doc/build_macosx.md).

----

## Problems: ##

We're building up a [Troubleshooting](https://github.com/Microsoft/malmo/wiki/Troubleshooting) page of the wiki for frequently encountered situations. If that doesn't work then please ask a question on our [chat page](https://gitter.im/Microsoft/malmo) or open a [new issue](https://github.com/Microsoft/malmo/issues/new).

----

## Launching Minecraft with our Mod: ##

Minecraft needs to create windows and render to them with OpenGL, so the machine you do this from must have a desktop environment.

Go to the folder where you unzipped the release, then:

`cd Minecraft`  
`launchClient` (On Windows)  
`./launchClient.sh` (On Linux or MacOSX)

or, e.g. `launchClient -port 10001` to launch Minecraft on a specific port.

on Linux or MacOSX: `./launchClient.sh -port 10001`

*NB: If you run this from a terminal, the bottom line will say something like "Building 95%" - ignore this - don't wait for 100%! As long as a Minecraft game window has opened and is displaying the main menu, you are good to go.*

By default the Mod chooses port 10000 if available, and will search upwards for a free port if not, up to 11000.
The port chosen is shown in the Mod config page.

To change the port while the Mod is running, use the `portOverride` setting in the Mod config page.

The Mod and the agents use other ports internally, and will find free ones in the range 10000-11000 so if administering
a machine for network use these TCP ports should be open.

----

## Launch an agent: ##

#### Running a Python agent: ####

```
cd Python_Examples
python3 run_mission.py
``` 

#### Running a C++ agent: ####

`cd Cpp_Examples`

To run the pre-built sample:

`run_mission` (on Windows)  
`./run_mission` (on Linux or MacOSX)

To build the sample yourself:

`cmake .`  
`cmake --build .`  
`./run_mission` (on Linux or MacOSX)  
`Debug\run_mission.exe` (on Windows)

#### Running a C# agent: ####

To run the pre-built sample (on Windows):

`cd CSharp_Examples`  
`CSharpExamples_RunMission.exe`

To build the sample yourself, open CSharp_Examples/RunMission.csproj in Visual Studio.

Or from the command-line:

`cd CSharp_Examples`

Then, on Windows:  
```
msbuild RunMission.csproj /p:Platform=x64
bin\x64\Debug\CSharpExamples_RunMission.exe
```

#### Running a Java agent: ####

`cd Java_Examples`  
`java -cp MalmoJavaJar.jar:JavaExamples_run_mission.jar -Djava.library.path=. JavaExamples_run_mission` (on Linux or MacOSX)  
`java -cp MalmoJavaJar.jar;JavaExamples_run_mission.jar -Djava.library.path=. JavaExamples_run_mission` (on Windows)

#### Running an Atari agent: (Linux only) ####

```
cd Python_Examples
python3 ALE_HAC.py
```

----

# Citations #

Please cite Malmo as:

Johnson M., Hofmann K., Hutton T., Bignell D. (2016) [_The Malmo Platform for Artificial Intelligence Experimentation._](http://www.ijcai.org/Proceedings/16/Papers/643.pdf) [Proc. 25th International Joint Conference on Artificial Intelligence](http://www.ijcai.org/Proceedings/2016), Ed. Kambhampati S., p. 4246. AAAI Press, Palo Alto, California USA. https://github.com/Microsoft/malmo

----

# Code of Conduct #

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.