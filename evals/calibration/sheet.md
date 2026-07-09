## v1-q06
**Provider:** google | **Mode:** bm25

### Question
What does control SC-7 (Boundary Protection) in SP 800-53 require organizations to monitor?

### Answer
Based on the provided excerpts, control SC-7 (Boundary Protection) requires organizations to monitor communications at the external managed interfaces to the system and at key internal managed interfaces within the system [3].

### Cited Excerpts
[3] sp800-53r5 §SC-7 — SC-7
```
BOUNDARY PROTECTION Control: a.Monitor and control communications at the external managed interfaces to the system and at key internal managed interfaces within the system; b.Implement subnetworks for publicly accessible system components that are [Selection: physically; logically] separated from internal organizational networks; and c. Connect to external networks or systems only through managed interfaces consisting of boundary protection devices arranged in accordance with an organizational security and privacy architecture. Discussion:  Managed interfaces include gateways, routers, firewalls, guards, network-based malicious code analysis, virtualization systems, or encrypted tunnels implemented within a security architecture. Subnetworks that are physically or logically separated from internal networks are referred to as demilitarized zones or DMZs. Restricting or prohibiting interfaces within organizational systems includes restricting external web traffic to designated web servers within managed interfaces, prohibiting external traffic that appears to be spoofing internal addresses, and prohibiting internal traffic that appears to be spoofing external addresses. [SP 800-189] provides additional information on source address validation techniques to prevent ingress and egress of traffic with spoofed addresses. Commercial telecommunications services are provided by network components and consolidated management systems shared by customers. These services may also include third party-provided access lines and other service elements. Such services may represent sources of increased risk despite contract security provisions. Boundary protection may be implemented as a common control for all or part of an organizational network such that the boundary to be protected is greater than a system-specific boundary (i.e., an authorization boundary). NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 298 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 Related Controls:  AC-4, AC-17, AC-18, AC-19, AC-20, AU-13, CA-3, CM-2, CM-4, CM-7, CM-10, CP- 8, CP-10, IR-4, MA-4, PE-3, PL-8, PM-12, SA-8, SA-17, SC-5, SC-26, SC-32, SC-35, SC-43.
```


### Label
```json
{
  "example_id": "v1-q06",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q07
**Provider:** anthropic | **Mode:** hybrid

### Question
What is the first requirement of control IA-5 (Authenticator Management) in SP 800-53?

### Answer
The first requirement of control IA-5 (Authenticator Management) in SP 800-53 is to verify the identity of the individual, group, role, service, or device receiving the authenticator as part of the initial authenticator distribution. Specifically, the control states: "Verifying, as part of the initial authenticator distribution, the identity of the individual, group, role, service, or device receiving the authenticator" [1].

### Cited Excerpts
[1] sp800-53r5 §IA-5 — IA-5
```
AUTHENTICATOR MANAGEMENT Control:  Manage system authenticators by: a.Verifying, as part of the initial authenticator distribution, the identity of the individual, group, role, service, or device receiving the authenticator; b.Establishing initial authenticator content for any authenticators issued by the organization; c. Ensuring that authenticators have sufficient strength of mechanism for their intended use; d.Establishing and implementing administrative procedures for initial authenticator distribution, for lost or compromised or damaged authenticators, and for revoking authenticators; e.Changing default authenticators prior to first use; f. Changing or refreshing authenticators [Assignment: organization-defined time period by authenticator type] or when [Assignment: organization-defined events] occur; g.Protecting authenticator content from unauthorized disclosure and modification; h.Requiring individuals to take, and having devices implement, specific controls to protect authenticators; and i. Changing authenticators for group or role accounts when membership to those accounts changes. Discussion:  Authenticators include passwords, cryptographic devices, biometrics, certificates, one-time password devices, and ID badges. Device authenticators include certificates and passwords. Initial authenticator content is the actual content of the authenticator (e.g., the initial password). In contrast, the requirements for authenticator content contain specific criteria or characteristics (e.g., minimum password length). Developers may deliver system components with factory default authentication credentials (i.e., passwords) to allow for initial installation and configuration. Default authentication credentials are often well known, easily discoverable, and present a significant risk. The requirement to protect individual authenticators may be implemented via control PL-4 or PS-6 for authenticators in the possession of individuals and by controls AC-3, AC-6, and SC-28 for authenticators stored in organizational systems, including passwords stored in hashed or encrypted formats or files containing encrypted or hashed passwords accessible with administrator privileges. Systems support authenticator management by organization-defined settings and restrictions for various authenticator characteristics (e.g., minimum password length, validation time window for NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 139 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 time synchronous one-time tokens, and number of allowed rejections during the verification stage of biometric authentication).
```


### Label
```json
{
  "example_id": "v1-q07",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q08
**Provider:** google | **Mode:** bm25

### Question
What is the primary purpose of control AU-2 (Event Logging) in SP 800-53?

### Answer
Based on the provided excerpts, the primary purpose of control AU-2 (Event Logging) is to identify, coordinate, specify, and review the types of events that a system is capable of logging in order to support the audit function, guide the selection criteria for events to be logged, and support after-the-fact investigations of incidents [3]. Specifically, the types of events that require logging are those that are significant and relevant to the security of systems and the privacy of individuals, as well as those that support specific monitoring and auditing needs [3].

### Cited Excerpts
[3] sp800-53r5 §AU-2 — AU-2
```
EVENT LOGGING Control: a.Identify the types of events that the system is capable of logging in support of the audit function: [Assignment: organization-defined event types that the system is capable of logging]; b.Coordinate the event logging function with other organizational entities requiring audit- related information to guide and inform the selection criteria for events to be logged; c. Specifythe following event types for logging within the system: [Assignment: organization- defined event types (subset of the event types defined in AU-2a.) along with the frequency of (or situation requiring) logging for each identified event type]; d.Provide a rationale for why the event types selected for logging are deemed to be adequate to support after-the-fact investigations of incidents; and e.Review and update the event types selected for logging [Assignment: organization-defined frequency]. Discussion:  An event is an observable occurrence in a system. The types of events that require logging are those events that are significant and relevant to the security of systems and the privacy of individuals. Event logging also supports specific monitoring and auditing needs. Event types include password changes, failed logons or failed accesses related to systems, security or privacy attribute changes, administrative privilege usage, PIV credential usage, data action changes, query parameters, or external credential usage. In determining the set of event types that require logging, organizations consider the monitoring and auditing appropriate for each of the controls to be implemented. For completeness, event logging includes all protocols that are operational and supported by the system. To balance monitoring and auditing requirements with other system needs, event logging requires identifying the subset of event types that are logged at a given point in time. For example, organizations may determine that systems need the capability to log every file access successful and unsuccessful, but not activate that capability except for specific circumstances due to the potential burden on system performance. The types of events that organizations desire to be logged may change. Reviewing and updating the set of logged events is necessary to help ensure that the events remain relevant and continue to support the needs of the organization. Organizations consider how the types of logging events can reveal information about individuals that may give rise to privacy risk and how best to mitigate such risks. For example, there is the potential to reveal personally identifiable information in the audit trail, especially if the logging event is based on patterns or time of usage. Event logging requirements, including the need to log specific event types, may be referenced in other controls and control enhancements.
```


### Label
```json
{
  "example_id": "v1-q08",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q09
**Provider:** google | **Mode:** dense

### Question
What must organizations develop according to control CP-1 in SP 800-53?

### Answer
According to control CP-1 in SP 800-53, organizations must develop, document, and disseminate to organization-defined personnel or roles:
* A contingency planning policy (at the organization-level, mission/business process-level, or system-level) that addresses purpose, scope, roles, responsibilities, management commitment, coordination among organizational entities, and compliance, and is consistent with applicable laws, executive orders, directives, regulations, policies, standards, and guidelines [4].
* Procedures to facilitate the implementation of the contingency planning policy and the associated contingency planning controls [4].

### Cited Excerpts
[4] sp800-53r5 §CP-1 — CP-1
```
POLICY AND PROCEDURES Control: a.Develop, document, and disseminate to [Assignment: organization-defined personnel or roles]: 1.[Selection (one or more): Organization-level; Mission/business process-level; System- level] contingency planning policy that: (a)Addresses purpose, scope, roles, responsibilities, management commitment, coordination among organizational entities, and compliance; and (b)Is consistent with applicable laws, executive orders, directives, regulations, policies, standards, and guidelines; and 2.Procedures to facilitate the implementation of the contingency planning policy and the associated contingency planning controls; b.Designate an [Assignment: organization-defined official] to manage the development, documentation, and dissemination of the contingency planning policy and procedures; and c. Review and update the current contingency planning: 1.Policy [Assignment: organization-defined frequency] and following [Assignment: organization-defined events]; and 2.Procedures [Assignment: organization-defined frequency] and following [Assignment: organization-defined events]. Discussion:  Contingency planning policy and procedures address the controls in the CP family that are implemented within systems and organizations. The risk management strategy is an important factor in establishing such policies and procedures. Policies and procedures contribute to security and privacy assurance. Therefore, it is important that security and privacy programs collaborate on the development of contingency planning policy and procedures. Security and privacy program policies and procedures at the organization level are preferable, in general, and may obviate the need for mission- or system-specific policies and procedures. The policy can be included as part of the general security and privacy policy or be represented by multiple policies that reflect the complex nature of organizations. Procedures can be established for security and privacy programs, for mission or business processes, and for systems, if needed. Procedures describe how the policies or controls are implemented and can be directed at the individual or role that is the object of the procedure. Procedures can be documented in system security and privacy plans or in one or more separate documents. Events that may precipitate an update to contingency planning policy and procedures include assessment or audit findings, security incidents or breaches, or changes in laws, executive orders, directives, regulations, policies, standards, and guidelines. Simply restating controls does not constitute an organizational policy or procedure. Related Controls:  PM-9, PS-8, SI-12. Control Enhancements:  None.
```


### Label
```json
{
  "example_id": "v1-q09",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q13
**Provider:** ollama | **Mode:** bm25

### Question
How do SP 800-53 controls AU-2 and AU-3 support the audit and accountability function?

### Answer
The SP 800-53 controls AU-2 and AU-3 support the audit and accountability function by requiring organizations to:

* Identify the types of events that the system is capable of logging in support of the audit function, including event types such as password changes, failed logons or accesses related to systems, security or privacy attribute changes, administrative privilege usage, PIV credential usage, data action changes, query parameters, or external credential usage [6].
* Coordinate the event logging function with other organizational entities requiring audit-related information to guide and inform the selection criteria for events to be logged [6].
* Specify the types of events that require logging within the system, including a subset of the event types defined in AU-2a, along with the frequency of (or situation requiring) logging for each identified event type [6].

These controls ensure that organizations have a clear understanding of what events are being logged and why, which is essential for maintaining an accurate audit trail and supporting after-the-fact investigations of incidents.

### Cited Excerpts
[6] sp800-53r5 §AU-2 — AU-2
```
EVENT LOGGING Control: a.Identify the types of events that the system is capable of logging in support of the audit function: [Assignment: organization-defined event types that the system is capable of logging]; b.Coordinate the event logging function with other organizational entities requiring audit- related information to guide and inform the selection criteria for events to be logged; c. Specifythe following event types for logging within the system: [Assignment: organization- defined event types (subset of the event types defined in AU-2a.) along with the frequency of (or situation requiring) logging for each identified event type]; d.Provide a rationale for why the event types selected for logging are deemed to be adequate to support after-the-fact investigations of incidents; and e.Review and update the event types selected for logging [Assignment: organization-defined frequency]. Discussion:  An event is an observable occurrence in a system. The types of events that require logging are those events that are significant and relevant to the security of systems and the privacy of individuals. Event logging also supports specific monitoring and auditing needs. Event types include password changes, failed logons or failed accesses related to systems, security or privacy attribute changes, administrative privilege usage, PIV credential usage, data action changes, query parameters, or external credential usage. In determining the set of event types that require logging, organizations consider the monitoring and auditing appropriate for each of the controls to be implemented. For completeness, event logging includes all protocols that are operational and supported by the system. To balance monitoring and auditing requirements with other system needs, event logging requires identifying the subset of event types that are logged at a given point in time. For example, organizations may determine that systems need the capability to log every file access successful and unsuccessful, but not activate that capability except for specific circumstances due to the potential burden on system performance. The types of events that organizations desire to be logged may change. Reviewing and updating the set of logged events is necessary to help ensure that the events remain relevant and continue to support the needs of the organization. Organizations consider how the types of logging events can reveal information about individuals that may give rise to privacy risk and how best to mitigate such risks. For example, there is the potential to reveal personally identifiable information in the audit trail, especially if the logging event is based on patterns or time of usage. Event logging requirements, including the need to log specific event types, may be referenced in other controls and control enhancements.
```


### Label
```json
{
  "example_id": "v1-q13",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q14
**Provider:** ollama | **Mode:** dense

### Question
How do controls SC-7 and SC-8 in SP 800-53 address system communications security?

### Answer
Controls SC-7 and SC-8 in SP 800-53 address system communications security by implementing cryptographic mechanisms to protect message externals, concealing or randomizing communication patterns, and using protected distribution systems to prevent unauthorized disclosure of information during transmission [1][2][5]. 

SC-7 is not mentioned as a separate control but rather as part of the discussion on SC-8.

### Cited Excerpts
[1] sp800-53r5 §SC-8 — SC-8
```
Control Enhancements: (1)TRANSMISSION CONFIDENTIALITY AND INTEGRITY |CRYPTOGRAPHIC PROTECTION Implement cryptographic mechanisms to [Selection (one or more): prevent unauthorized disclosure of information; detect changes to information] during transmission. Discussion:  Encryption protects information from unauthorized disclosure and modification during transmission. Cryptographic mechanisms that protect the confidentiality and integrity of information during transmission include TLS and IPSec. Cryptographic mechanisms used to protect information integrity include cryptographic hash functions that have applications in digital signatures, checksums, and message authentication codes. Related Controls:  SC-12,  SC-13. (2)TRANSMISSION CONFIDENTIALITY AND INTEGRITY |PRE- AND POST-TRANSMISSION HANDLING Maintain the [Selection (one or more): confidentiality; integrity] of information during preparation for transmission and during reception. Discussion:  Information can be unintentionally or maliciously disclosed or modified during preparation for transmission or during reception, including during aggregation, at protocol transformation points, and during packing and unpacking. Such unauthorized disclosures or modifications compromise the confidentiality or integrity of the information. Related Controls:  None. (3)TRANSMISSION CONFIDENTIALITY AND INTEGRITY |CRYPTOGRAPHIC PROTECTION FOR MESSAGE EXTERNALS Implement cryptographic mechanisms to protect message externals unless otherwise protected by [Assignment: organization-defined alternative physical controls]. Discussion:  Cryptographic protection for message externals addresses protection from the unauthorized disclosure of information. Message externals include message headers and routing information. Cryptographic protection prevents the exploitation of message externals and applies to internal and external networks or links that may be visible to individuals who are not authorized users. Header and routing information is sometimes transmitted in clear text (i.e., unencrypted) because the information is not identified by organizations as having significant value or because encrypting the information can result in lower network performance or higher costs. Alternative physical controls include protected distribution systems. Related Controls:  SC-12, SC-13.
```

[2] sp800-53r5 §SC-8 — SC-8
```
TRANSMISSION CONFIDENTIALITY AND INTEGRITY Control:  Protect the [Selection (one or more): confidentiality; integrity] of transmitted information. Discussion:  Protecting the confidentiality and integrity of transmitted information applies to internal and external networks as well as any system components that can transmit information, including servers, notebook computers, desktop computers, mobile devices, printers, copiers, scanners, facsimile machines, and radios. Unprotected communication paths are exposed to the possibility of interception and modification. Protecting the confidentiality and integrity of information can be accomplished by physical or logical means. Physical protection can be achieved by using protected distribution systems. A protected distribution system is a wireline or fiber-optics telecommunications system that includes terminals and adequate electromagnetic, NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 305 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 acoustical, electrical, and physical controls to permit its use for the unencrypted transmission of classified information. Logical protection can be achieved by employing encryption techniques. Organizations that rely on commercial providers who offer transmission services as commodity services rather than as fully dedicated services may find it difficult to obtain the necessary assurances regarding the implementation of needed controls for transmission confidentiality and integrity. In such situations, organizations determine what types of confidentiality or integrity services are available in standard, commercial telecommunications service packages. If it is not feasible to obtain the necessary controls and assurances of control effectiveness through appropriate contracting vehicles, organizations can implement appropriate compensating controls. Related Controls:  AC-17, AC-18, AU-10, IA-3, IA-8, IA-9, MA-4, PE-4, SA-4, SA-8, SC-7, SC-16, SC- 20, SC-23, SC-28. Control Enhancements: (1)TRANSMISSION CONFIDENTIALITY AND INTEGRITY |CRYPTOGRAPHIC PROTECTION Implement cryptographic mechanisms to [Selection (one or more): prevent unauthorized disclosure of information; detect changes to information] during transmission. Discussion:  Encryption protects information from unauthorized disclosure and modification during transmission. Cryptographic mechanisms that protect the confidentiality and integrity of information during transmission include TLS and IPSec. Cryptographic mechanisms used to protect information integrity include cryptographic hash functions that have applications in digital signatures, checksums, and message authentication codes.
```

[5] sp800-53r5 §SC-8 — SC-8
```
Transmission Confidentiality and Integrity S SC-8(1) CRYPTOGRAPHIC PROTECTION S SC-8(2) PRE- AND POST-TRANSMISSION HANDLING S SC-8(3) CRYPTOGRAPHIC PROTECTION FOR MESSAGE EXTERNALS S SC-8(4) CONCEAL OR RANDOMIZE COMMUNICATIONS S SC-8(5) PROTECTED DISTRIBUTION SYSTEM S Transmission Confidentiality W: Incorporated into SC-8. Trusted Path S √ SC-11(1) IRREFUTABLE COMMUNICATIONS PATH S √ Cryptographic Key Establishment and Management O/S SC-12(1) AVAILABILITY O/S SC-12(2) SYMMETRIC KEYS O/S SC-12(3) ASYMMETRIC KEYS O/S SC-12(4) PKI CERTIFICATES W: Incorporated into SC-12(3). SC-12(5) PKI CERTIFICATES / HARDWARE TOKENS W: Incorporated into SC-12(3). SC-12(6) PHYSICAL CONTROL OF KEYS O/S Cryptographic Protection S SC-13(1) FIPS-VALIDATED CRYPTOGRAPHY W: Incorporated into SC-13. SC-13(2) NSA-APPROVED CRYPTOGRAPHY W: Incorporated into SC-13. SC-13(3) INDIVIDUALS WITHOUT FORMAL ACCESS APPROVALS W: Incorporated into SC-13. SC-13(4) DIGITAL SIGNATURES W: Incorporated into SC-13. Public Access Protections W: Incorporated into AC-2, AC-3, AC-5, SI- 3, SI-4, SI-5, SI-7, and SI-10. Collaborative Computing Devices and Applications S SC-15(1) PHYSICAL OR LOGICAL DISCONNECT S SC-15(2) BLOCKING INBOUND AND OUTBOUND COMMUNICATIONS TRAFFIC W: Incorporated into SC-7. SC-15(3) DISABLING AND REMOVAL IN SECURE WORK AREAS O SC-15(4) EXPLICITLY INDICATE CURRENT PARTICIPANTS S Transmission of Security and Privacy Attributes S SC-16(1) INTEGRITY VERIFICATION S SC-16(2) ANTI-SPOOFING MECHANISMS S SC-16(3) CRYPTOGRAPHIC BINDING S
```


### Label
```json
{
  "example_id": "v1-q14",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q16
**Provider:** ollama | **Mode:** hybrid

### Question
According to SP 800-171, how are system account management (3.1.1) and access restrictions (3.1.2) related?

### Answer
According to SP 800-171, system account management (AC-02) and access restrictions (AC-03) are related in that they both contribute to ensuring proper control over who has access to the system. AC-02 requires organizations to define, document, approve, and enforce physical and logical access restrictions associated with changes to the system [7]. This is closely tied to AC-03, which requires organizations to define, document, approve, and enforce physical and logical access controls for accessing the system [7].

Additionally, both AC-02 and AC-03 are part of the Access Control (AC) family of controls in SP 800-171. The Tailoring Criteria in Appendix C also lists AC-01 Policy and Procedures, AC-02 Account Management, and AC-03 Access Enforcement as related to Controlled Unclassified Information (CUI) [8].

Therefore, system account management and access restrictions are closely linked in ensuring that only authorized individuals have access to the system.

### Cited Excerpts
[7] sp800-53r5 §CM-5 — CM-5
```
ACCESS RESTRICTIONS FOR CHANGE Control:  Define, document, approve, and enforce physical and logical access restrictions associated with changes to the system. Discussion:  Changes to the hardware, software, or firmware components of systems or the operational procedures related to the system can potentially have significant effects on the security of the systems or individuals’ privacy. Therefore, organizations permit only qualified and authorized individuals to access systems for purposes of initiating changes. Access restrictions include physical and logical access controls (see AC-3 and PE-3), software libraries, workflow automation, media libraries, abstract layers (i.e., changes implemented into external interfaces rather than directly into systems), and change windows (i.e., changes occur only during specified times). Related Controls:  AC-3, AC-5, AC-6, CM-9, PE-3, SC-28, SC-34, SC-37, SI-2, SI-10. Control Enhancements: (1)ACCESS RESTRICTIONS FOR CHANGE |AUTOMATED ACCESS ENFORCEMENT AND AUDIT RECORDS (a)Enforce access restrictions using [Assignment: organization-defined automated mechanisms]; and (b)Automatically generate audit records of the enforcement actions. Discussion:  Organizations log system accesses associated with applying configuration changes to ensure that configuration change control is implemented and to support after- the-fact actions should organizations discover any unauthorized changes. Related Controls:  AU-2, AU-6, AU-7, AU-12, CM-6, CM-11, SI-12. (2)ACCESS RESTRICTIONS FOR CHANGE | REVIEW SYSTEM CHANGES [Withdrawn: Incorporated into CM-3(7).] (3)ACCESS RESTRICTIONS FOR CHANGE | SIGNED COMPONENTS [Withdrawn: Moved to CM-14.] (4)ACCESS RESTRICTIONS FOR CHANGE |DUAL AUTHORIZATION Enforce dual authorization for implementing changes to [Assignment: organization- defined system components and system-level information]. Discussion:  Organizations employ dual authorization to help ensure that any changes to selected system components and information cannot occur unless two qualified individuals approve and implement such changes. The two individuals possess the skills and expertise to determine if the proposed changes are correct implementations of approved changes. The individuals are also accountable for the changes. Dual authorization may also be known as two-person control. To reduce the risk of collusion, organizations consider rotating dual authorization duties to other individuals. System-level information includes operational procedures.
```

[8] sp800-171r3 §appendix-c — Appendix C.Tailoring Criteria
```
Access Control (AC) NIST SP 800-53 CONTROLS MODERATE BASELINE TAILORING CRITERIA SECURITY REQUIREMENT AC-01 Policy and Procedures CUI 03.15.01 AC-02 Account Management CUI 03.01.01 AC-02(01) Account Management | Automated System Account Management NCO — AC-02(02) Account Management | Automated Temporary and Emergency Account Management NCO — AC-02(03) Account Management | Disable Accounts CUI 03.01.01 AC-02(04) Account Management | Automated Audit Actions NCO — AC-02(05) Account Management | Inactivity Logout CUI 03.01.01 AC-02(13) Account Management |Disable Accounts for High-Risk Individuals CUI 03.01.01 AC-03 Access Enforcement CUI 03.01.02 AC-04 Information Flow Enforcement CUI 03.01.03 AC-05 Separation of Duties CUI 03.01.04 AC-06 Least Privilege CUI 03.01.05 AC-06(01) Least Privilege |Authorize Access to Security Functions CUI 03.01.05 NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 94 NIST SP 800-53 CONTROLS MODERATE BASELINE TAILORING CRITERIA SECURITY REQUIREMENT AC-06(02) Least Privilege | Non-Privileged Access for Non-Security Functions CUI 03.01.06 AC-06(05) Least Privilege| Privileged Accounts CUI 03.01.06 AC-06(07) Least Privilege | Review of User Privileges CUI 03.01.05 AC-06(09) Least Privilege| Log Use of Privileged Functions CUI 03.01.07 AC-06(10) Least
```


### Label
```json
{
  "example_id": "v1-q16",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q22
**Provider:** anthropic | **Mode:** bm25

### Question
How do the SP 800-53 control families relate to the minimum security requirements of FIPS 200?

### Answer
Based on the excerpts, the relationship between SP 800-53 control families and FIPS 200 minimum security requirements can be described as follows:

Of the 20 control families in SP 800-53, **17 are aligned with the minimum security requirements in FIPS 200** [1]. The three families that are *not* aligned with FIPS 200 are:

- **Program Management (PM)** — addresses enterprise-level program management [1]
- **PII Processing and Transparency (PT)** — addresses privacy considerations [1]
- **Supply Chain Risk Management (SR)** — addresses supply chain risk considerations [1]

These three families "address enterprise-level program management, privacy, and supply chain risk considerations pertaining to federal mandates emergent since [FIPS 200]," meaning they were developed to address requirements that arose after FIPS 200 was published [1].

Additionally, FIPS 200 defines a **Security Control Baseline** as "the set of minimum security controls defined for a low-impact, moderate-impact, or high-impact information system," and **Security Controls** as "the management, operational, and technical controls (i.e., safeguards or countermeasures) prescribed for an information system to protect the confidentiality, integrity, and availability of the system and its information" [5]. SP 800-53 provides the catalog of controls from which these baselines are drawn to meet those minimum requirements.

### Cited Excerpts
[1] sp800-53r5 §2.2 — 2.2CONTROLSTRUCTUREANDORGANIZATION
```
Security and privacy controls described in this publication have a well-defined organization and structure. For ease of use in the security and privacy control selection and specification process, controls are organized into 20 families.25 Each family contains controls that are related to the specific topic of the family. A two-character identifier uniquely identifies each control family (e.g., PS for Personnel Security). Security and privacy controls may involve aspects of policy, oversight, supervision, manual processes, and automated mechanisms that are implemented by systems or actions by individuals. Table 1 lists the security and privacy control families and their associated family identifiers. TABLE 1: SECURITY AND PRIVACY CONTROL FAMILIES ID FAMILY ID FAMILY AC Access Control PE Physical and Environmental Protection AT Awareness and Training PL Planning AU Audit and Accountability PM Program Management CA Assessment, Authorization, and Monitoring PS Personnel Security CM Configuration Management PT PII Processing and Transparency CP Contingency Planning RA Risk Assessment IA Identification and Authentication SA System and Services Acquisition IR Incident Response SC System and Communications Protection MA Maintenance SI System and Information Integrity MP Media Protection SR Supply Chain Risk Management Families of controls contain base controls and control enhancements, which are directly related to their base controls. Control enhancements either add functionality or specificity to a base control or increase the strength of a base control. Control enhancements are used in systems and environments of operation that require greater protection than the protection provided by the base control. The need for organizations to select and implement control enhancements is due to the potential adverse organizational or individual impacts or when organizations require additions to the base control functionality or assurance based on assessments of risk. The 25 Of the 20 control families in NIST SP 800-53, 17 are aligned with the minimum security requirements in [FIPS 200]. The Program Management (PM), PII Processing and Transparency (PT), and Supply Chain Risk Management (SR) families address enterprise-level program management, privacy, and supply chain risk considerations pertaining to federal mandates emergent since [FIPS 200]. NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER TWO PAGE 9 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 selection and implementation of control enhancements always requires the selection and implementation of the base control. The families are arranged in alphabetical order, while the controls and control enhancements within each family are in numerical order.
```

[5] fips-200 §appendix-a — APPENDIX A     TERMS AND DEFINITIONS
```
SAFEGUARDS:  Protective measures prescribed to meet the security requirements (i.e., confidentiality, integrity, and availability) specified for an information system. Safeguards may include security features, management constraints, personnel security, and security of physical structures, areas, and devices.  [CNSS Instruction 4009 Adapted]Synonymous with security controls and countermeasures. SANITIZATION:Process to remove information from media such that information recovery is not possible. It includes removing all labels, markings, and activity logs.  [CNSS Instruction 4009 Adapted] 8 FIPS Publication 200                              Minimum Security Requirements for Federal Information and Information Systems ________________________________________________________________________________________________ SECURITY CATEGORY:  The characterization of information or an information system based on an assessment of the potential impact that a loss of confidentiality, integrity, or availability of such information or information system would have on organizational operations, organizational assets, or individuals.  [FIPS Publication 199] SECURITY CONTROLS:  The management, operational, and technical controls (i.e., safeguards or countermeasures) prescribed for an information system to protect the confidentiality, integrity, and availability of the system and its information.  [FIPS Publication 199] SECURITY CONTROL BASELINE:  The set of minimum security controls defined for a low-impact, moderate-impact, or high-impact information system. SECURITY OBJECTIVE:  Confidentiality, integrity, or availability.  [FIPS Publication 199] SECURITY PLAN:  See System Security Plan. SECURITY REQUIREMENTS:  Requirements levied on an information system that are derived from applicable laws, Executive Orders, directives, policies, standards, instructions, regulations, or procedures, or organizational mission/business case needs to ensure the confidentiality, integrity, and availability of the information being processed, stored, or transmitted. SENIOR AGENCY INFORMATION SECURITY OFFICER:  Official responsible for carrying out the Chief Information Officer responsibilities under FISMA and serving as the Chief Information Officer’s primary liaison to the agency’s authorizing officials, information system owners, and information system security officers.  [44 U.S.C., Sec. 3544] SYSTEM:  See information system. SYSTEM SECURITY PLAN:Formal document that provides an overview of the security requirements for an information system and describes the security controls in place or planned for meeting those requirements.
```


### Label
```json
{
  "example_id": "v1-q22",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q23
**Provider:** google | **Mode:** hybrid

### Question
How do the account management requirements in SP 800-171 (3.1.1) and SP 800-53 (AC-2) differ in their scope and application?

### Answer
Based on the provided excerpts, the account management requirements in SP 800-171 (03.01.01) and SP 800-53 (AC-2) differ in their scope, structure, and specific application in the following ways:

* **Source Control Consolidation:** SP 800-171 requirement 03.01.01 consolidates the core requirements of SP 800-53 control AC-2 along with several of its control enhancements, specifically AC-2(03) (Disable Accounts), AC-2(05) (Inactivity Logout), and AC-2(13) (Disable Accounts for High-Risk Individuals) into a single requirement [5]. In contrast, SP 800-53 separates these into the base AC-2 control and distinct, individual control enhancements [7][8].
* **Account Types and Prohibitions:** SP 800-53 AC-2 requires organizations to assign account managers, require organization-defined prerequisites for group/role membership, and establish a process for changing shared or group account authenticators when individuals are removed [2]. SP 800-171's 03.01.01 does not include these specific administrative mandates [3]. Additionally, SP 800-53 lists "shared" accounts among the types of accounts that may be prohibited or require authenticator changes [2], whereas SP 800-171's list of potentially prohibited risk-associated accounts includes group, emergency, guest, anonymous, and temporary accounts, but does not explicitly list "shared" accounts [3][5].
* **Conditions for Disabling Accounts:** Under SP 800-171 03.01.01, accounts must be disabled when they have expired, been inactive for an organization-defined period, are no longer associated with a user, are in violation of policy, or when "significant risks associated with individuals are discovered" [3]. In SP 800-53, these actions are split: the first four conditions are mapped to control enhancement AC-2(3) [8], while disabling accounts due to significant risks (high-risk individuals) is mapped to a separate enhancement, AC-2(13) [6].
* **Inactivity Logout:** SP 800-171 incorporates the inactivity logout requirement directly as part of its main 03.01.01 requirement (item h) [3], whereas SP 800-53 places this requirement under the separate control enhancement AC-2(5) [8].

### Cited Excerpts
[5] sp800-171r3 §03.01.01 — 03.01.01   Account Management
```
Types of accounts that organizations may prohibit due to increased risk include group, emergency, guest, anonymous, and temporary. Organizations may choose to define access privileges or other attributes by account, type of account, or a combination of both. Other attributes required for authorizing access include restrictions on the time of day, day of the week, and point of origin. When defining other system account attributes, organizations consider system requirements (e.g., system upgrades, scheduled maintenance) and mission and business requirements (e.g., time zone differences, remote access to facilitate travel requirements). Users who pose a significant security risk include individuals for whom reliable evidence indicates either the intention to use authorized access to the system to cause harm or that adversaries will cause harm through them. Close coordination NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 8 among mission and business owners, system administrators, human resource managers, and legal staff is essential when disabling system accounts for high-risk individuals. Time periods for the notification of organizational personnel or roles may vary. Inactivity logout is behavior- or policy-based and requires users to take physical action to log out when they are expecting inactivity longer than the defined period. Automatic enforcement of inactivity logout is addressed by 03.01.10. REFERENCES Source Controls: AC-02, AC-02(03), AC-02(05), AC-02(13) Supporting Publications: SP 800-46 [14], SP 800-57-1 [15], SP 800-57-2 [16], SP 800- 57-3 [17], SP 800-77 [18], SP 800-113 [19], SP 800-114 [20], SP 800-121 [21], SP 800- 162 [22], SP 800-178 [23], SP 800-192 [24], IR 7874 [25], IR 7966 [26]
```

[7] sp800-53r5 §AC-2 — AC-2
```
Account Management O AC-2(1) AUTOMATED SYSTEM ACCOUNT MANAGEMENT O AC-2(2) AUTOMATED TEMPORARY AND EMERGENCY ACCOUNT MANAGEMENT S AC-2(3) DISABLE ACCOUNTS S AC-2(4) AUTOMATED AUDIT ACTIONS S AC-2(5) INACTIVITY LOGOUT O/S AC-2(6) DYNAMIC PRIVILEGE MANAGEMENT S AC-2(7) PRIVILEGED USER ACCOUNTS O AC-2(8) DYNAMIC ACCOUNT MANAGEMENT S AC-2(9) RESTRICTIONS ON USE OF SHARED AND GROUP ACCOUNTS O AC-2(10) SHARED AND GROUP ACCOUNT CREDENTIAL CHANGE W: Incorporated into AC-2k. AC-2(11) USAGE CONDITIONS S AC-2(12) ACCOUNT MONITORING FOR ATYPICAL USAGE O/S AC-2(13) DISABLE ACCOUNTS FOR HIGH-RISK INDIVIDUALS O Access Enforcement S AC-3(1) RESTRICTED ACCESS TO PRIVILEGED FUNCTIONS W: Incorporated into AC-6. AC-3(2) DUAL AUTHORIZATION S AC-3(3) MANDATORY ACCESS CONTROL S AC-3(4) DISCRETIONARY ACCESS CONTROL S AC-3(5) SECURITY-RELEVANT INFORMATION S AC-3(6) PROTECTION OF USER AND SYSTEM INFORMATION W: Incorporated into MP-4 and SC-28. AC-3(7) ROLE-BASED ACCESS CONTROL O/S AC-3(8) REVOCATION OF ACCESS AUTHORIZATIONS O/S AC-3(9) CONTROLLED RELEASE O/S AC-3(10) AUDITED OVERRIDE OF ACCESS CONTROL MECHANISMS O AC-3(11) RESTRICT ACCESS TO SPECIFIC INFORMATION TYPES S AC-3(12) ASSERT AND ENFORCE APPLICATION ACCESS S AC-3(13) ATTRIBUTE-BASED ACCESS CONTROL S AC-3(14) INDIVIDUAL ACCESS S AC-3(15) DISCRETIONARY AND MANDATORY ACCESS CONTROL S
```

[8] sp800-53r5 §AC-2 — AC-2
```
Automated mechanisms can include internal system functions and email, telephonic, and text messaging notifications. Related Controls:  None. (2)ACCOUNT MANAGEMENT |AUTOMATED TEMPORARY AND EMERGENCY ACCOUNT MANAGEMENT Automatically [Selection: remove; disable] temporary and emergency accounts after [Assignment: organization-defined time period for each type of account]. Discussion:  Management of temporary and emergency accounts includes the removal or disabling of such accounts automatically after a predefined time period rather than at the convenience of the system administrator. Automatic removal or disabling of accounts provides a more consistent implementation. Related Controls:  None. (3)ACCOUNT MANAGEMENT |DISABLE ACCOUNTS Disable accounts within [Assignment: organization-defined time period] when the accounts: NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 21 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 (a)Have expired; (b)Are no longer associated with a user or individual; (c)Are in violation of organizational policy; or (d)Have been inactive for [Assignment: organization-defined time period]. Discussion:  Disabling expired, inactive, or otherwise anomalous accounts supports the concepts of least privilege and least functionality which reduce the attack surface of the system. Related Controls:  None. (4)ACCOUNT MANAGEMENT |AUTOMATED AUDIT ACTIONS Automatically audit account creation, modification, enabling, disabling, and removal actions. Discussion:  Account management audit records are defined in accordance with AU-2 and reviewed, analyzed, and reported in accordance with AU-6. Related Controls:  AU-2, AU-6. (5)ACCOUNT MANAGEMENT |INACTIVITY LOGOUT Require that users log out when [Assignment: organization-defined time period of expected inactivity or description of when to log out]. Discussion:  Inactivity logout is behavior- or policy-based and requires users to take physical action to log out when they are expecting inactivity longer than the defined period. Automatic enforcement of inactivity logout is addressed by AC-11. Related Controls:  AC-11. (6)ACCOUNT MANAGEMENT |DYNAMIC PRIVILEGE MANAGEMENT Implement [Assignment: organization-defined dynamic privilege management capabilities].
```

[2] sp800-53r5 §AC-2 — AC-2
```
ACCOUNT MANAGEMENT Control: a.Define and document the types of accounts allowed and specifically prohibited for use within the system; b.Assign account managers; c. Require [Assignment: organization-defined prerequisites and criteria] for group and role membership; d.Specify: 1.Authorized users of the system; 2.Group and role membership; and 3.Access authorizations (i.e., privileges) and [Assignment: organization-defined attributes (as required)] for each account; e.Require approvals by [Assignment: organization-defined personnel or roles] for requests to create accounts; f. Create, enable, modify, disable, and remove accounts in accordance with [Assignment: organization-defined policy, procedures, prerequisites, and criteria]; g.Monitor the use of accounts; h.Notify account managers and [Assignment: organization-defined personnel or roles] within: 1.[Assignment: organization-defined time period] when accounts are no longer required; 2.[Assignment: organization-defined time period] when users are terminated or transferred; and 3.[Assignment: organization-defined time period] when system usage or need-to-know changes for an individual; i. Authorize access to the system based on: 1.A valid access authorization; 2.Intended system usage; and 3.[Assignment: organization-defined attributes (as required)]; j. Review accounts for compliance with account management requirements [Assignment: organization-defined frequency]; k. Establish and implement a process for changing shared or group account authenticators (if deployed) when individuals are removed from the group; and l. Align account management processes with personnel termination and transfer processes. Discussion:  Examples of system account types include individual, shared, group, system, guest, anonymous, emergency, developer, temporary, and service. Identification of authorized system users and the specification of access privileges reflect the requirements in other controls in the security plan. Users requiring administrative privileges on system accounts receive additional scrutiny by organizational personnel responsible for approving such accounts and privileged access, including system owner, mission or business owner, senior agency information security officer, or senior agency official for privacy. Types of accounts that organizations may wish to prohibit due to increased risk include shared, group, emergency, anonymous, temporary, and guest accounts.
```

[3] sp800-171r3 §03.01.01 — 03.01.01   Account Management
```
a.Define the types of system accounts allowed and prohibited. b.Create, enable, modify, disable, and remove system accounts in accordance with policy, procedures, prerequisites, and criteria. c.Specify: 1.Authorized users of the system, 2.Group and role membership, and 3.Access authorizations (i.e., privileges) for each account. d.Authorize access to the system based on: 1.A valid access authorization and 2.Intended system usage. e.Monitor the use of system accounts. NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 7 f.Disable system accounts when: 1.The accounts have expired, 2.The accounts have been inactive for [Assignment: organization-defined time period], 3.The accounts are no longer associated with a user or individual, 4.The accounts are in violation of organizational policy, or 5.Significant risks associated with individuals are discovered. g.Notify account managers and designated personnel or roles within: 1.[Assignment: organization-defined time period] when accounts are no longer required. 2.[Assignment: organization-defined time period] when users are terminated or transferred. 3.[Assignment: organization-defined time period] when system usage or the need-to-know changes for an individual. h.Require that users log out of the system after [Assignment: organization-defined time period] of expected inactivity or when [Assignment: organization-defined circumstances]. DISCUSSION This requirement focuses on account management for systems and applications. The definition and enforcement of access authorizations other than those determined by account type (e.g., privileged access, non-privileged access) are addressed in 03.01.02. System account types include individual, group, temporary, system, guest, anonymous, emergency, developer, and service. Users who require administrative privileges on system accounts receive additional scrutiny by personnel responsible for approving such accounts and privileged access. Types of accounts that organizations may prohibit due to increased risk include group, emergency, guest, anonymous, and temporary. Organizations may choose to define access privileges or other attributes by account, type of account, or a combination of both. Other attributes required for authorizing access include restrictions on the time of day, day of the week, and point of origin. When defining other system account attributes, organizations consider system requirements (e.g., system upgrades, scheduled maintenance) and mission and business requirements (e.g., time zone differences, remote access to facilitate travel requirements).
```

[6] sp800-53r5 §AC-2 — AC-2
```
Discussion:  Specifying and enforcing usage conditions helps to enforce the principle of least privilege, increase user accountability, and enable effective account monitoring. Account monitoring includes alerts generated if the account is used in violation of organizational parameters. Organizations can describe specific conditions or circumstances under which system accounts can be used, such as by restricting usage to certain days of the week, time of day, or specific durations of time. Related Controls:  None. (12)ACCOUNT MANAGEMENT |ACCOUNT MONITORING FOR ATYPICAL USAGE (a)Monitor system accounts for [Assignment: organization-defined atypical usage]; and (b)Report atypical usage of system accounts to [Assignment: organization-defined personnel or roles]. Discussion:  Atypical usage includes accessing systems at certain times of the day or from locations that are not consistent with the normal usage patterns of individuals. Monitoring for atypical usage may reveal rogue behavior by individuals or an attack in progress. Account monitoring may inadvertently create privacy risks since data collected to identify atypical usage may reveal previously unknown information about the behavior of individuals. Organizations assess and document privacy risks from monitoring accounts for atypical NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 23 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 usage in their privacy impact assessment and make determinations that are in alignment with their privacy program plan. Related Controls:  AU-6, AU-7, CA-7, IR-8, SI-4. (13)ACCOUNT MANAGEMENT |DISABLE ACCOUNTS FOR HIGH-RISK INDIVIDUALS Disable accounts of individuals within [Assignment: organization-defined time period] of discovery of [Assignment: organization-defined significant risks]. Discussion:  Users who pose a significant security and/or privacy risk include individuals for whom reliable evidence indicates either the intention to use authorized access to systems to cause harm or through whom adversaries will cause harm. Such harm includes adverse impacts to organizational operations, organizational assets, individuals, other organizations, or the Nation. Close coordination among system administrators, legal staff, human resource managers, and authorizing officials is essential when disabling system accounts for high-risk individuals. Related Controls:  AU-6, SI-4. References:  [SP 800-162], [SP 800-178], [SP 800-192].
```


### Label
```json
{
  "example_id": "v1-q23",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q25
**Provider:** ollama | **Mode:** hybrid

### Question
How does security categorization defined in FIPS 199 inform the control baseline selection requirements in FIPS 200?

### Answer
Security categorization defined in FIPS 199 informs the control baseline selection requirements in FIPS 200 by providing a framework to categorize information systems based on their potential impact on national security or other critical functions, which in turn guides the selection of controls from the moderate baseline in SP 800-53 [1]. The tailoring actions for RA-2 (Security Categorization) in SP 800-53r5 indicate that organizations must perform an impact-level prioritization and a supply chain risk assessment to inform their control baseline selection requirements [7].

### Cited Excerpts
[1] sp800-171r3 §03.17.03 — 03.17.03    Supply Chain Requirements and Processes
```
Available at https://www.govinfo.gov/app/details/DCPD-200901022 [3]Atomic Energy Act (P.L. 83-703), August 1954. Available at https://www.govinfo.gov/app/details/STATUTE-68/STATUTE-68-Pg919 [4]National Archives and Records Administration (2019) Controlled Unclassified Information (CUI) Registry. Available at https://www.archives.gov/cui [5]32 CFR Part 2002 (2016), Controlled Unclassified Information (CUI), September 2016. Available at https://www.govinfo.gov/content/pkg/CFR-2018-title32-vol6/pdf/CFR-2018- title32-vol6-part2002.pdf [6]National Institute of Standards and Technology (2004) Standards for Security Categorization of Federal Information and Information Systems. (U.S. Department of Commerce, Washington, DC), Federal Information Processing Standards Publication (FIPS) 199. https://doi.org/10.6028/NIST.FIPS.199 [7]National Institute of Standards and Technology (2006) Minimum Security Requirements for Federal Information and Information Systems. (U.S. Department of Commerce, Washington, DC), Federal Information Processing Standards Publication (FIPS) 200. https://doi.org/10.6028/NIST.FIPS.200 [8]Joint Task Force (2020) Security and Privacy Controls for Information Systems and Organizations. (National Institute of Standards and Technology, Gaithersburg, MD), NIST Special Publication (SP) 800-53, Rev. 5, Includes updates as of December 10, 2020. https://doi.org/10.6028/NIST.SP.800-53r5 [9]Federal Information Security Modernization Act (P.L. 113-283), December 2014. Available at https://www.govinfo.gov/app/details/PLAW-113publ283 [10]Ross RS, Pillitteri VY, Graubart R, Bodeau D, McQuaid R (2021) Developing Cyber-Resilient Systems: A Systems Security Engineering Approach. (National Institute of Standards and Technology, Gaithersburg, MD), NIST Special Publication (SP) 800-160, Vol. 2, Rev. 1.
```

[7] sp800-53r5 §RA-2 — RA-2
```
Security Categorization O RA-2(1) IMPACT-LEVEL PRIORITIZATION O Risk Assessment O √ RA-3(1) SUPPLY CHAIN RISK ASSESSMENT O √ RA-3(2) USE OF ALL-SOURCE INTELLIGENCE O √ RA-3(3) DYNAMIC THREAT AWARENESS O √ RA-3(4) PREDICTIVE CYBER ANALYTICS O √ Vulnerability Monitoring and Scanning O √ RA-5(1) UPDATE TOOL CAPABILITY W: Incorporated into RA-5. RA-5(2) UPDATE VULNERABILITIES TO BE SCANNED O √ RA-5(3) BREADTH AND DEPTH OF COVERAGE O √ RA-5(4) DISCOVERABLE INFORMATION O √ RA-5(5) PRIVILEGED ACCESS O √ RA-5(6) AUTOMATED TREND ANALYSES O √ RA-5(7) AUTOMATED DETECTION AND NOTIFICATION OF UNAUTHORIZED COMPONENTS W: Incorporated into CM-8. RA-5(8) REVIEW HISTORIC AUDIT LOGS O √ RA-5(9) PENETRATION TESTING AND ANALYSES W: Incorporated into CA-8. RA-5(10) CORRELATE SCANNING INFORMATION O √ RA-5(11) PUBLIC DISCLOSURE PROGRAM O √ Threat Hunting O/S √ NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ APPENDIX C PAGE 454 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 TABLE C-17:  SYSTEM AND SERVICES ACQUISITION FAMILY CONTROL NUMBER CONTROL NAME CONTROL ENHANCEMENT NAME IMPLEMENTED BY ASSURANCE
```


### Label
```json
{
  "example_id": "v1-q25",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q26
**Provider:** ollama | **Mode:** hybrid

### Question
What specific password length and complexity requirements does NIST recommend as best practices for federal systems?

### Answer
NIST recommends that organizations enforce composition and complexity rules for passwords, which are organization-defined. This means that the specific password length and complexity requirements can vary depending on the organization's needs and policies.

[6] sp800-171r3 §03.05.07 — 03.05.07 Password Management (p.45)

### Cited Excerpts
[6] sp800-171r3 §03.05.07 — 03.05.07    Password Management
```
a.Maintain a list of commonly-used, expected, or compromised passwords, and update the list [Assignment: organization-defined frequency] and when organizational passwords are suspected to have been compromised. b.Verify that passwords are not found on the list of commonly used, expected, or compromised passwords when users create or update passwords. c.Transmit passwords only over cryptographically protected channels. d.Store passwords in a cryptographically protected form. e.Select a new password upon first use after account recovery. f.Enforce the following composition and complexity rules for passwords: [Assignment: organization-defined composition and complexity rules]. DISCUSSION Password-based authentication applies to passwords used in single-factor or multi- factor authentication. Long passwords or passphrases are preferable to shorter passwords. Enforced composition rules provide marginal security benefits while decreasing usability. However, organizations may choose to establish and enforce certain rules for password generation (e.g., minimum character length) under certain circumstances. For example, account recovery can occur when a password is forgotten. Cryptographically protected passwords include salted one-way cryptographic hashes of passwords. The list of commonly used, compromised, or expected passwords includes passwords obtained from previous breach corpuses, dictionary words, and repetitive or sequential characters. The list includes context- specific words, such as the name of the service, username, and derivatives thereof. Changing temporary passwords to permanent passwords immediately after system logon ensures that the necessary strength of the authentication mechanism is implemented at the earliest opportunity and reduces susceptibility to authenticator compromises. Long passwords and passphrases can be used to increase the complexity of passwords. NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 37 REFERENCES Source Control: IA-05(01) Supporting Publications: SP 800-63-3 [27] Obscure feedback of authentication information during the authentication process. DISCUSSION Authentication feedback does not provide information that would allow unauthorized individuals to compromise authentication mechanisms. For example, for desktop or notebook systems with relatively large monitors, the threat may be significant (commonly referred to as shoulder surfing). For mobile devices with small displays, this threat may be less significant and is balanced against the increased likelihood of input errors due to small keyboards. Therefore, the means of obscuring authenticator feedback is selected accordingly. Obscuring feedback includes displaying asterisks when users type passwords into input devices or displaying feedback for a limited time before fully obscuring it. REFERENCES Source Control: IA-06 Supporting Publications: None
```


### Label
```json
{
  "example_id": "v1-q26",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v1-q27
**Provider:** ollama | **Mode:** bm25

### Question
Which specific security tools or products does NIST recommend for implementing boundary protection controls?

### Answer
Host-based firewalls are mentioned as part of boundary protection controls, specifically in the context of limiting external network connections to the system [5.3] and implementing subnetworks for publicly accessible system components that are physically or logically separated from internal networks [7.a]. 

Additionally, gateways, routers, firewalls, network-based malicious code analysis, virtualization systems, and encrypted tunnels implemented within a security architecture are also mentioned as examples of managed interfaces [7.c].

It's worth noting that the excerpts do not provide an exhaustive list of specific security tools or products recommended by NIST for implementing boundary protection controls.

### Label
```json
{
  "example_id": "v1-q27",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q32
**Provider:** ollama | **Mode:** hybrid

### Question
What does MANAGE 1.3 in the NIST AI RMF require regarding responses to high-priority AI risks?

### Answer
Responses to high-priority AI risks, as identified by the MAP function, are developed, planned, and documented. Risk response options can include mitigating, transferring, avoiding, or accepting [1].

### Cited Excerpts
[1] ai-rmf §manage-1-3 — MANAGE 1.3: Responses to the AI risks deemed high priority, as
```
identified by theMAP function, are developed, planned, and doc- umented.Risk response options can include mitigating, transfer- ring, avoiding, or accepting. MANAGE1.4:Negative residual risks (defined as the sum of all unmitigatedrisks)tobothdownstreamacquirersofAIsystems and end users are documented. MANAGE 2: Strategies to maximize AI benefits and minimize negative impacts are planned, prepared, implemented, documented, and informed by input from relevant AI actors. MANAGE2.1:Resources required to manage AI risks are taken into account – along with viable non-AI alternative systems, ap- proaches, or methods – to reduce the magnitude or likelihood of potential impacts. MANAGE2.2:Mechanismsareinplaceandappliedtosustain the value of deployed AI systems.
```


### Label
```json
{
  "example_id": "v2-q32",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q33
**Provider:** ollama | **Mode:** dense

### Question
According to the NIST AI RMF, what does GOVERN 1.4 specify as the foundation for establishing a risk management process, and what does GOVERN 1.6 require regarding resource allocation?

### Answer
According to GOVERN 1.4, the risk management process and its outcomes are established through transparent policies, procedures, and other controls based on organizational risk priorities [4].

There is no mention of GOVERN 1.6 in the provided excerpts.

### Cited Excerpts
[4] ai-rmf §govern-1-4 — GOVERN 1.4: The risk management process and its outcomes are
```
establishedthroughtransparentpolicies,procedures,andother controls based on organizational risk priorities. Categories Subcategories Continued on next page Page 22 NIST AI 100-1 AI RMF 1.0 Table 1:Categories and subcategories for theGOVERN function.(Continued) GOVERN1.5:Ongoingmonitoringandperiodicreviewofthe riskmanagementprocessanditsoutcomesareplannedandor- ganizationalroles andresponsibilities clearlydefined,including determining the frequency of periodic review.
```


### Label
```json
{
  "example_id": "v2-q33",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q34
**Provider:** anthropic | **Mode:** bm25

### Question
How do the MAP and MEASURE functions in the NIST AI RMF differ in their focus on AI system evaluation?

### Answer
The MAP and MEASURE functions differ in their focus as follows:

The **MAP function** is oriented toward *identifying and characterizing* AI risks, contexts, and impacts. It defines the specific tasks and methods the AI system will support [7], documents knowledge limits and human oversight considerations [7], identifies internal risk controls for AI system components [8], and characterizes the likelihood and magnitude of beneficial and harmful impacts [8]. In essence, MAP establishes the landscape of risks and contexts that will inform subsequent evaluation.

The **MEASURE function**, by contrast, employs *quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts* [8]. It "uses knowledge relevant to AI risks identified in the MAP function and informs the MANAGE function" [8]. Specifically, MEASURE focuses on testing AI systems before deployment and regularly while in operation, tracking metrics for trustworthy characteristics, social impact, and human-AI configurations, and producing rigorous software testing and performance assessment with associated measures of uncertainty and formalized reporting [8]. It also evaluates privacy risk, fairness and bias, environmental impact, and the effectiveness of TEVV metrics [3].

A key distinction is that MAP *identifies and documents* risks and impacts, while MEASURE *applies methods and metrics to quantify, track, and validate* those risks [2][8]. Furthermore, "where tradeoffs among the trustworthy characteristics arise, measurement provides a traceable basis to inform management decisions" [8] — a role that MAP does not fulfill. After MEASURE is complete, "objective, repeatable, or scalable test, evaluation, verification, and validation (TEVV) processes including metrics, methods, and methodologies are in place, followed, and documented" [2], whereas MAP's output serves as the foundational input to that process.

### Cited Excerpts
[7] ai-rmf §map-2-1 — MAP 2.1:The specific tasks and methods used to implement the
```
tasks that the AI system will support are defined (e.g., classifiers, generative models, recommenders). MAP2.2:InformationabouttheAIsystem’sknowledgelimits and how system output may be utilized and overseen by humans isdocumented.Documentationprovidessufficientinformation toassistrelevantAIactorswhenmakingdecisionsandtaking subsequent actions. Categories Subcategories Continued on next page Page 26 NIST AI 100-1 AI RMF 1.0 Table 2:Categories and subcategories for theMAP function.(Continued)
```

[8] ai-rmf §map-4-1 — MAP 4.1: Approaches for mapping AI technology and legal risks
```
of its components – including the use of third-party data or soft- ware – are in place, followed, and documented, as are risks of in- fringement of a third party’s intellectual property or other rights. MAP4.2:InternalriskcontrolsforcomponentsoftheAIsys- tem,includingthird-partyAItechnologies,areidentifiedand documented. MAP 5:Impacts to individuals, groups, communities, organizations, and society are characterized. MAP5.1:Likelihoodandmagnitudeofeachidentifiedimpact (both potentially beneficial and harmful) based on expected use, pastusesofAIsystemsinsimilarcontexts,publicincidentre- ports,feedbackfromthoseexternaltotheteamthatdeveloped ordeployedtheAIsystem,orotherdataareidentifiedand documented. Categories Subcategories Continued on next page Page 27 NIST AI 100-1 AI RMF 1.0 Table 2:Categories and subcategories for theMAP function.(Continued) MAP5.2:Practicesandpersonnelforsupportingregularen- gagement with relevant AI actors and integrating feedback about positive,negative,andunanticipatedimpactsareinplaceand documented. Categories Subcategories 5.3 Measure TheMEASUREfunctionemploysquantitative,qualitative,ormixed-methodtools,tech- niques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts.It uses knowledge relevant to AI risks identified in theMAP function and informs theMANAGEfunction.AIsystemsshouldbetestedbeforetheirdeploymentandregu- larlywhileinoperation.AIriskmeasurementsincludedocumentingaspectsofsystems’ functionality and trustworthiness. Measuring AI risks includes tracking metrics for trustworthy characteristics, social impact, and human-AI configurations.Processes developed or adopted in theMEASUREfunction should include rigorous software testing and performance assessment methodologies with associated measures of uncertainty, comparisons to performance benchmarks, and formal- ized reporting and documentation of results. Processes for independent review can improve the effectiveness of testing and can mitigate internal biases and potential conflicts of inter- est. Where tradeoffs among the trustworthy characteristics arise, measurement provides a trace- ablebasistoinformmanagementdecisions.Optionsmayincluderecalibration,impact mitigation, or removal of the system from design, development, production, or use, as well as a range of compensating, detective, deterrent, directive, and recovery controls.
```

[3] ai-rmf §measure-2-9 — MEASURE 2.9:The AI model is explained, validated, and docu-
```
mented, and AI system output is interpreted within its context – as identified in theMAP function – to inform responsible use and governance. MEASURE2.10:Privacy risk of the AI system – as identified in theMAP function – is examined and documented. MEASURE2.11:Fairnessandbias–asidentifiedintheMAP function – are evaluated and results are documented. MEASURE2.12:Environmental impact and sustainability of AI modeltrainingandmanagementactivities–asidentifiedinthe MAP function – are assessed and documented. MEASURE2.13:EffectivenessoftheemployedTEVVmet- ricsandprocessesintheMEASUREfunctionareevaluatedand documented. MEASURE 3: Mechanisms for tracking identified AI risks over time are in place. MEASURE3.1:Approaches,personnel,anddocumentationare inplacetoregularlyidentifyandtrackexisting,unanticipated, and emergent AI risks based on factors such as intended and ac- tual performance in deployed contexts. MEASURE3.2:Risktrackingapproachesareconsideredfor settingswhereAIrisksaredifficulttoassessusingcurrently availablemeasurementtechniquesorwheremetricsarenotyet available. Categories Subcategories Continued on next page Page 30 NIST AI 100-1 AI RMF 1.0 Table 3:Categories and subcategories for theMEASURE function.(Continued) MEASURE3.3:Feedback processes for end users and impacted communities to report problems and appeal system outcomes are established and integrated into AI system evaluation metrics. MEASURE 4: Feedback about efficacy of measurement is gathered and assessed.
```

[2] ai-rmf §map-4-1 — MAP 4.1: Approaches for mapping AI technology and legal risks
```
Processes for independent review can improve the effectiveness of testing and can mitigate internal biases and potential conflicts of inter- est. Where tradeoffs among the trustworthy characteristics arise, measurement provides a trace- ablebasistoinformmanagementdecisions.Optionsmayincluderecalibration,impact mitigation, or removal of the system from design, development, production, or use, as well as a range of compensating, detective, deterrent, directive, and recovery controls. After completing the MEASURE function, objective, repeatable, or scalable test, evaluation, verification, and validation (TEVV) processes including metrics, methods, and methodolo- giesareinplace,followed,anddocumented.Metricsandmeasurementmethodologies should adhere to scientific, legal, and ethical norms and be carried out in an open and trans- parentprocess.Newtypesofmeasurement,qualitativeandquantitative,mayneedtobe developed.The degree to which each measurement type provides unique and meaningful information to the assessment of AI risks should be considered.Framework users will en- hance their capacity to comprehensively evaluate system trustworthiness, identify and track existing and emergent risks, and verify efficacy of the metrics. Measurement outcomes will be utilized in theMANAGE function to assist risk monitoring and response efforts.It is in- cumbent on Framework users to continue applying theMEASUREfunction to AI systems as knowledge, methodologies, risks, and impacts evolve over time. Page 28 NIST AI 100-1 AI RMF 1.0 Practices related to measuring AI risks are described in the NIST AI RMF Playbook.Table 3 lists theMEASURE function’s categories and subcategories. Table 3:Categories and subcategories for theMEASURE function. MEASURE 1: Appropriate methods and metrics are identified and applied. MEASURE1.1:Approaches and metrics for measurement of AI risks enumerated during the MAP function are selected for imple- mentationstartingwiththemostsignificantAIrisks.Therisks ortrustworthinesscharacteristicsthatwillnot–orcannot–be measured are properly documented.
```


### Label
```json
{
  "example_id": "v2-q34",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q39
**Provider:** ollama | **Mode:** hybrid

### Question
What security objectives does FIPS 199 define, and what account management requirements does SP 800-53 control AC-2 specify?

### Answer
FIPS 199 defines security objectives, which include confidentiality, integrity, and availability. 

SP 800-53 control AC-2 specifies account management requirements, including:

* Identifying authorized system users and specifying access privileges [8]
* Establishing conditions for disabling or deactivating accounts, such as when shared/group, emergency, or temporary accounts are no longer required, or when individuals are transferred or terminated [8]

These requirements reflect the need to balance security with operational needs, ensuring that access is granted only to authorized personnel and that accounts are properly managed to prevent unauthorized access.

### Cited Excerpts
[8] sp800-53r5 §AC-2 — AC-2
```
Discussion:  Examples of system account types include individual, shared, group, system, guest, anonymous, emergency, developer, temporary, and service. Identification of authorized system users and the specification of access privileges reflect the requirements in other controls in the security plan. Users requiring administrative privileges on system accounts receive additional scrutiny by organizational personnel responsible for approving such accounts and privileged access, including system owner, mission or business owner, senior agency information security officer, or senior agency official for privacy. Types of accounts that organizations may wish to prohibit due to increased risk include shared, group, emergency, anonymous, temporary, and guest accounts. NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 20 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 Where access involves personally identifiable information, security programs collaborate with the senior agency official for privacy to establish the specific conditions for group and role membership; specify authorized users, group and role membership, and access authorizations for each account; and create, adjust, or remove system accounts in accordance with organizational policies. Policies can include such information as account expiration dates or other factors that trigger the disabling of accounts. Organizations may choose to define access privileges or other attributes by account, type of account, or a combination of the two. Examples of other attributes required for authorizing access include restrictions on time of day, day of week, and point of origin. In defining other system account attributes, organizations consider system-related requirements and mission/business requirements. Failure to consider these factors could affect system availability. Temporary and emergency accounts are intended for short-term use. Organizations establish temporary accounts as part of normal account activation procedures when there is a need for short-term accounts without the demand for immediacy in account activation. Organizations establish emergency accounts in response to crisis situations and with the need for rapid account activation. Therefore, emergency account activation may bypass normal account authorization processes. Emergency and temporary accounts are not to be confused with infrequently used accounts, including local logon accounts used for special tasks or when network resources are unavailable (may also be known as accounts of last resort). Such accounts remain available and are not subject to automatic disabling or removal dates. Conditions for disabling or deactivating accounts include when shared/group, emergency, or temporary accounts are no longer required and when individuals are transferred or terminated.
```


### Label
```json
{
  "example_id": "v2-q39",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q40
**Provider:** google | **Mode:** bm25

### Question
According to SP 800-171 Revision 3, what must organizations provide regarding security literacy training?

### Answer
According to SP 800-171 Revision 3, organizations must provide security literacy training to system users (including managers, senior executives, system administrators, and contractors) [6]. Specifically, organizations must provide this training:

1. As part of initial training for new users and at an organization-defined frequency thereafter [6];
2. When required by system changes or following organization-defined events [6]; and
3. On recognizing and reporting indicators of insider threat, social engineering, and social mining [6]. 

Additionally, organizations must update the security literacy training content at an organization-defined frequency and following organization-defined events [6].

### Cited Excerpts
[6] sp800-171r3 §03.02.01 — 03.02.01    Literacy Training and Awareness
```
a.Provide security literacy training to system users: 1.As part of initial training for new users and [Assignment: organization- defined frequency] thereafter, 2.When required by system changes or following [Assignment: organization- defined events], and 3.On recognizing and reporting indicators of insider threat, social engineering, and social mining. b.Update security literacy training content [Assignment: organization-defined frequency] and following [Assignment: organization-defined events]. DISCUSSION Organizations provide basic and advanced levels of security literacy training to system users (including managers, senior executives, system administrators, and contractors) and measures to test the knowledge level of users. Organizations determine the content of literacy training based on specific organizational requirements, the systems to which personnel have authorized access, and work environments (e.g., telework). The content includes an understanding of the need for security and the actions required of users to maintain security and respond to incidents. The content also addresses the need for operations security and the handling of CUI. Security awareness techniques include displaying posters, offering supplies inscribed with security reminders, generating email advisories or notices from organizational officials, displaying logon screen messages, and conducting awareness events using podcasts, videos, and webinars. Security literacy training is conducted at a frequency consistent with applicable laws, directives, regulations, and policies. Updating literacy training content on a regular basis ensures that the content remains relevant. Events that may precipitate an update to literacy training content include NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 20 assessment or audit findings, security incidents or breaches, or changes in applicable laws, Executive Orders, directives, regulations, policies, standards, and guidelines. Potential indicators and possible precursors of insider threats include behaviors such as inordinate, long-term job dissatisfaction; attempts to gain access to information that is not required for job performance; unexplained access to financial resources; sexual harassment or bullying of fellow employees; workplace violence; and other serious violations of the policies, procedures, rules, directives, or practices of organizations. Organizations may consider tailoring insider threat awareness topics to roles (e.g., training for managers may be focused on specific changes in the behavior of team members, while training for employees may be focused on more general observations). Social engineering is an attempt to deceive an individual into revealing information or taking an action that can be used to breach, compromise, or otherwise adversely impact a system.
```


### Label
```json
{
  "example_id": "v2-q40",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q44
**Provider:** ollama | **Mode:** hybrid

### Question
What does FIPS 200 define as security requirements, and what are the initial training and frequency requirements in SP 800-171 Revision 3?

### Answer
FIPS 200 defines minimum security requirements for federal information and information systems, specifically standards for security categorization. 

In SP 800-171 Revision 3, organizations are required to provide initial training in the employment and operation of physical security controls to organization-defined personnel or roles at an organization-defined frequency [6]. Additionally, they must provide literacy training to system users as part of initial training for new users and at an organization-defined frequency thereafter, when required by system changes or following organization-defined events, and on recognizing and reporting indicators of insider threat, social engineering, and social mining [8].

### Cited Excerpts
[6] sp800-53r5 §AT-3 — AT-3
```
(2)ROLE-BASED TRAINING |PHYSICAL SECURITY CONTROLS Provide [Assignment: organization-defined personnel or roles] with initial and [Assignment: organization-defined frequency] training in the employment and operation of physical security controls. Discussion:  Physical security controls include physical access control devices, physical intrusion and detection alarms, operating procedures for facility security guards, and monitoring or surveillance equipment. Related Controls:  PE-2, PE-3, PE-4. (3)ROLE-BASED TRAINING |PRACTICAL EXERCISES Provide practical exercises in security and privacy training that reinforce training objectives. Discussion:  Practical exercises for security include training for software developers that addresses simulated attacks that exploit common software vulnerabilities or spear or whale phishing attacks targeted at senior leaders or executives. Practical exercises for privacy include modules with quizzes on identifying and processing personally identifiable information in various scenarios or scenarios on conducting privacy impact assessments. Related Controls:  None. (4)ROLE-BASED TRAINING | SUSPICIOUS COMMUNICATIONS AND ANOMALOUS SYSTEM BEHAVIOR [Withdrawn: Moved to AT-2(4)]. (5)ROLE-BASED TRAINING |PROCESSING PERSONALLY IDENTIFIABLE INFORMATION Provide [Assignment: organization-defined personnel or roles]with initial and [Assignment: organization-defined frequency] training in the employment and operation of personally identifiable information processing and transparency controls. Discussion:  Personally identifiable information processing and transparency controls include the organization’s authority to process personally identifiable information and personally identifiable information processing purposes. Role-based training for federal agencies addresses the types of information that may constitute personally identifiable information and the risks, considerations, and obligations associated with its processing. Such training also considers the authority to process personally identifiable information documented in privacy policies and notices, system of records notices, computer matching agreements and NISTSP800-53,REV.5SECURITY AND PRIVACY CONTROLS FOR INFORMATION SYSTEMS AND ORGANIZATIONS _________________________________________________________________________________________________ CHAPTER THREE PAGE 64 This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-53r5 notices, privacy impact assessments, [PRIVACT] statements, contracts, information sharing agreements, memoranda of understanding, and/or other documentation. Related Controls:  PT-2, PT-3, PT-5, PT-6. References:  [OMB A-130], [SP 800-50], [SP 800-181].
```

[8] sp800-171r3 §03.02.01 — 03.02.01    Literacy Training and Awareness
```
a.Provide security literacy training to system users: 1.As part of initial training for new users and [Assignment: organization- defined frequency] thereafter, 2.When required by system changes or following [Assignment: organization- defined events], and 3.On recognizing and reporting indicators of insider threat, social engineering, and social mining. b.Update security literacy training content [Assignment: organization-defined frequency] and following [Assignment: organization-defined events]. DISCUSSION Organizations provide basic and advanced levels of security literacy training to system users (including managers, senior executives, system administrators, and contractors) and measures to test the knowledge level of users. Organizations determine the content of literacy training based on specific organizational requirements, the systems to which personnel have authorized access, and work environments (e.g., telework). The content includes an understanding of the need for security and the actions required of users to maintain security and respond to incidents. The content also addresses the need for operations security and the handling of CUI. Security awareness techniques include displaying posters, offering supplies inscribed with security reminders, generating email advisories or notices from organizational officials, displaying logon screen messages, and conducting awareness events using podcasts, videos, and webinars. Security literacy training is conducted at a frequency consistent with applicable laws, directives, regulations, and policies. Updating literacy training content on a regular basis ensures that the content remains relevant. Events that may precipitate an update to literacy training content include NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 20 assessment or audit findings, security incidents or breaches, or changes in applicable laws, Executive Orders, directives, regulations, policies, standards, and guidelines. Potential indicators and possible precursors of insider threats include behaviors such as inordinate, long-term job dissatisfaction; attempts to gain access to information that is not required for job performance; unexplained access to financial resources; sexual harassment or bullying of fellow employees; workplace violence; and other serious violations of the policies, procedures, rules, directives, or practices of organizations. Organizations may consider tailoring insider threat awareness topics to roles (e.g., training for managers may be focused on specific changes in the behavior of team members, while training for employees may be focused on more general observations). Social engineering is an attempt to deceive an individual into revealing information or taking an action that can be used to breach, compromise, or otherwise adversely impact a system.
```


### Label
```json
{
  "example_id": "v2-q44",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q46
**Provider:** anthropic | **Mode:** dense

### Question
What three security objectives does FIPS 199 define, and what insider threat recognition training does SP 800-171 Revision 3 require?

### Answer
The three security objectives defined by FIPS 199 are **confidentiality, integrity, and availability** [1]. These are referenced in FIPS 200's definitions, where a low-impact system is described as one "in which all three security objectives (i.e., confidentiality, integrity, and availability) are assigned a FIPS 199 potential impact value of low" [1].

Regarding insider threat recognition training in SP 800-171 Revision 3: the tailoring criteria table shows that control **AT-02(02) Literacy Training and Awareness | Insider Threat** is mapped to security requirement **CUI 03.02.01** [6]. This indicates that insider threat awareness training is required under that security requirement. However, the excerpts do not provide the specific text or detailed language of requirement CUI 03.02.01 itself — only the mapping is shown [6].

Therefore, while the three FIPS 199 security objectives can be fully answered [1], the specific wording of what insider threat recognition training SP 800-171r3 requires (i.e., the actual requirement text of CUI 03.02.01) is not stated in the provided excerpts [6].

[NO_ANSWER] The excerpts identify that AT-02(02) maps to CUI 03.02.01 but do not provide the actual text of that security requirement describing what insider threat recognition training is specifically required.

### Cited Excerpts
[1] fips-200 §appendix-a — APPENDIX A     TERMS AND DEFINITIONS
```
[FIPS Publication 199] INTEGRITY:  Guarding against improper information modification or destruction, and includes ensuring information non-repudiation and authenticity.  [44 U.S.C., SEC. 3542] LOW-IMPACTSYSTEM:An information system in which all three security objectives (i.e., confidentiality, integrity, and availability) are assigned a FIPS 199 potential impact value of low. MANAGEMENT CONTROLS:  The security controls (i.e., safeguards or countermeasures) for an information system that focus on the management of risk and the management of information system security. 7 FIPS Publication 200                              Minimum Security Requirements for Federal Information and Information Systems ________________________________________________________________________________________________ MEDIA:Physical devices or writing surfaces including, but not limited to, magnetic tapes, optical disks, magnetic disks, Large-Scale Integration (LSI) memory chips, printouts (but not including display media) onto which information is recorded, stored, or printed within an information system. MODERATE-IMPACT SYSTEM: An information system in which at least one security objective (i.e., confidentiality, integrity, or availability) is assigned a FIPS 199 potential impact value of moderate, and no security objective is assigned a FIPS 199 potential impact value of high. NATIONAL SECURITY INFORMATION:  Information that has been determined pursuant to Executive Order 12958 as amended by Executive Order 13292, or any predecessor order, or by the Atomic Energy Act of 1954, as amended, to require protection against unauthorized disclosure and is marked to indicate its classified status. NATIONAL SECURITY SYSTEM:  Any information system (including any telecommunications system) used or operated by an agency or by a contractor of an agency, or other organization on behalf of an agency— (i) the function, operation, or use of which involves intelligence activities; involves cryptologic activities related to national security; involves command and control of military forces; involves equipment that is an integral part of a weapon or weapons system; or is critical to the direct fulfillment of military or intelligence missions (excluding a system that is to be used for routine administrative and business applications, for example, payroll, finance, logistics, and personnel management applications); or (ii) is protected at all times by procedures established for information that have been specifically authorized under criteria established by an Executive Order or an Act of Congress to be kept classified in the interest of national defense or foreign policy.  [44 U.S.C., SEC.
```

[6] sp800-171r3 §appendix-c — Appendix C.Tailoring Criteria
```
Awareness and Training (AT) NIST SP 800-53 CONTROLS MODERATE BASELINE TAILORING CRITERIA SECURITY REQUIREMENT AT-01 Policy and Procedures CUI 03.15.01 AT-02 Literacy Training and Awareness CUI 03.02.01 AT-02(02) Literacy Training and Awareness | Insider Threat CUI 03.02.01 AT-02(03) Literacy Training and Awareness | Social Engineering and Mining CUI 03.02.01 AT-03 Role-Based Training CUI 03.02.02 NIST SP 800-171r3 Protecting Controlled Unclassified Information May 2024 95 NIST SP 800-53 CONTROLS MODERATE BASELINE TAILORING CRITERIA SECURITY REQUIREMENT AT-04 Training Records NCO — Table 5. Audit and Accountability (AU) NIST SP 800-53 CONTROLS MODERATE BASELINE TAILORING CRITERIA SECURITY REQUIREMENT AU-01 Policy and Procedures CUI 03.15.01 AU-02 Event Logging CUI 03.03.01 AU-03 Content of Audit Records CUI 03.03.02 AU-03(01) Additional Audit Information CUI 03.03.02 AU-04 Audit Log Storage Capacity NCO — AU-05 Response to Audit Logging Process Failures CUI 03.03.04 AU-06 Audit Record Review, Analysis, and Reporting CUI 03.03.05 AU-06(01) Audit Record Review, Analysis, and Reporting | Automated Process Integration NCO — AU-06(03) Audit Record Review, Analysis, and Reporting | Correlate Audit Record Repositories CUI 03.03.05 AU-07 Audit Record Reduction and Report Generation CUI 03.03.06 AU-07(01) Audit Record Reduction and Report Generation | Automatic Processing NCO — AU-08 Time Stamps CUI 03.03.07 AU-09 Protection of Audit Information CUI 03.03.08 AU-09(04) Protection of Audit Information | Access by Subset of Privileged Users CUI 03.03.08 AU-11 Audit Record Retention CUI 03.03.03 AU-12 Audit Record Generation CUI 03.03.03 Table 6.
```


### Label
```json
{
  "example_id": "v2-q46",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```

## v2-q49
**Provider:** ollama | **Mode:** dense

### Question
What is the cost-benefit analysis framework that NIST recommends for selecting between different SP 800-53 control baselines?

### Answer
The cost-benefit analysis framework that NIST recommends for selecting between different SP 800-53 control baselines is not explicitly stated in the provided excerpts. However, it can be inferred from excerpt [2] that organizations should consider mission and business requirements, stakeholder protection needs, threats, vulnerabilities, and requirements to comply with federal laws, executive orders, directives, regulations, policies, standards, and guidelines when selecting a control baseline.

A more precise answer is:
Organizations must select a control baseline based on the needs of stakeholders, considering mission and business requirements as well as mandates imposed by applicable laws, executive orders, directives, policies, regulations, standards, and guidelines [2].

### Cited Excerpts
[2] sp800-53r5 §PL-10 — PL-10
```
BASELINE SELECTION Control:  Select a control baseline for the system. Discussion:  Control baselines are predefined sets of controls specifically assembled to address the protection needs of a group, organization, or community of interest. Controls are chosen for baselines to either satisfy mandates imposed by laws, executive orders, directives, regulations, policies, standards, and guidelines or address threats common to all users of the baseline under the assumptions specific to the baseline. Baselines represent a starting point for the protection of individuals’ privacy, information, and information systems with subsequent tailoring actions to manage risk in accordance with mission, business, or other constraints (see PL-11). Federal control baselines are provided in [SP 800-53B]. The selection of a control baseline is determined by the needs of stakeholders. Stakeholder needs consider mission and business requirements as well as mandates imposed by applicable laws, executive orders, directives, policies, regulations, standards, and guidelines. For example, the control baselines in [SP 800-53B] are based on the requirements from [FISMA] and [PRIVACT]. The requirements, along with the NIST standards and guidelines implementing the legislation, direct organizations to select one of the control baselines after the reviewing the information types and the information that is processed, stored, and transmitted on the system; analyzing the potential adverse impact of the loss or compromise of the information or system on the organization’s operations and assets, individuals, other organizations, or the Nation; and considering the results from system and organizational risk assessments. [CNSSI 1253] provides guidance on control baselines for national security systems. Related Controls:  PL-2, PL-11, RA-2, RA-3, SA-8. Control Enhancements:  None. References:  [FIPS 199], [FIPS 200], [SP 800-30], [SP 800-37], [SP 800-39], [SP 800-53B], [SP 800- 60-1], [SP 800-60-2], [SP 800-160-1], [CNSSI 1253].
```


### Label
```json
{
  "example_id": "v2-q49",
  "labeler": "",
  "faithfulness": null,
  "relevance": null,
  "citation_accuracy": null,
  "notes": ""
}
```
